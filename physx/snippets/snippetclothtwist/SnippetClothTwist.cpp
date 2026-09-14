// PhysX GPU deformable-surface reproduction of Newton's cloth_twist_xpbd scene.
// Newton is used only by export_mesh.py to export the reference rest mesh.
#include "PxPhysicsAPI.h"
#include "extensions/PxDeformableSurfaceExt.h"
#include "extensions/PxCudaHelpersExt.h"
#include "cudamanager/PxCudaContext.h"
#include "../snippetrender/SnippetRender.h"
#include "../snippetrender/SnippetCamera.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>

using namespace physx;

namespace
{
struct ErrorCallback : PxDefaultErrorCallback
{
    std::atomic<bool> failed{false};
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
    {
        if (code != PxErrorCode::eDEBUG_INFO && code != PxErrorCode::eDEBUG_WARNING)
            failed = true;
        PxDefaultErrorCallback::reportError(code, message, file, line);
    }
};
PxDefaultAllocator gAllocator;
ErrorCallback gErrors;
PxFoundation* gFoundation = NULL;
PxPhysics* gPhysics = NULL;
PxCudaContextManager* gCuda = NULL;
PxDefaultCpuDispatcher* gDispatcher = NULL;
PxScene* gScene = NULL;
PxDeformableSurface* gCloth = NULL;
PxTriangleMesh* gMesh = NULL;
PxDeformableSurfaceMaterial* gMaterial = NULL;
PxVec4* gPositions = NULL;
PxVec4* gVelocities = NULL;
PxVec4* gRest = NULL;
std::vector<PxVec3> gVertices;
std::vector<PxReal> gMasses;
std::vector<PxU32> gTriangles;
std::vector<PxU32> gAnchors;
std::vector<float> gSigns;
Snippets::Camera* gCamera = NULL;
const char* gDumpPath = NULL;
FILE* gDump = NULL;
const char* gMeshPath = "square_cloth.mesh";
unsigned gFrame = 0;
unsigned gFrames = 600;
unsigned gSubsteps = 10;
unsigned gIterations = 12;
bool gHeadless = false;
bool gBenchmark = false;
bool gPaused = false;
bool gSelfContact = true;
bool gDat = true;
bool gWire = false;
bool gExtensions = false;
bool gValid = true;
const float gFrameDt = 1.0f / 60.0f;

bool loadMesh()
{
    std::ifstream in(gMeshPath);
    unsigned vertices = 0, triangles = 0;
    if (!(in >> vertices >> triangles) || vertices < 4 || vertices > 1000000 ||
        triangles < 2 || triangles > 2000000)
    {
        std::fprintf(stderr, "Cannot read mesh %s; run export_mesh.py first.\n", gMeshPath);
        return false;
    }
    gVertices.resize(vertices);
    gMasses.resize(vertices);
    gTriangles.resize(3 * triangles);
    float minY = PX_MAX_F32, maxY = -PX_MAX_F32;
    for (unsigned i = 0; i < vertices; ++i)
    {
        PxVec3& v = gVertices[i];
        if (!(in >> v.x >> v.y >> v.z >> gMasses[i]) || !v.isFinite() ||
            !std::isfinite(gMasses[i]) || gMasses[i] <= 0)
            return false;
        minY = PxMin(minY, v.y);
        maxY = PxMax(maxY, v.y);
    }
    for (unsigned& i : gTriangles)
        if (!(in >> i) || i >= vertices)
            return false;
    for (unsigned i = 0; i < vertices; ++i)
    {
        const float y = gVertices[i].y;
        if (PxAbs(y - minY) < 1.e-6f || PxAbs(y - maxY) < 1.e-6f)
        {
            gAnchors.push_back(i);
            gSigns.push_back(PxAbs(y - maxY) < 1.e-6f ? 1.f : -1.f);
        }
    }
    std::printf("Mesh: %u vertices, %u triangles, %zu anchors; Y=[%.3f, %.3f]\n",
        vertices, triangles, gAnchors.size(), minY, maxY);
    return !gAnchors.empty() && maxY > minY;
}

float bendingMaterial(float thickness)
{
    struct Edge { float area = 0.f; unsigned faces = 0; };
    std::map<std::pair<unsigned, unsigned>, Edge> edges;
    for (unsigned t = 0; t < gTriangles.size(); t += 3)
    {
        const PxVec3& a = gVertices[gTriangles[t]];
        const PxVec3& b = gVertices[gTriangles[t + 1]];
        const PxVec3& c = gVertices[gTriangles[t + 2]];
        const float area = 0.5f * (b - a).cross(c - a).magnitude();
        for (unsigned e = 0; e < 3; ++e)
        {
            const unsigned i = gTriangles[t + e], j = gTriangles[t + (e + 1) % 3];
            Edge& edge = edges[std::make_pair(PxMin(i, j), PxMax(i, j))];
            edge.area += area;
            ++edge.faces;
        }
    }
    // Newton's reference hinge coefficient is 1e-3 * rest edge length.
    // PhysX cooks B * thickness^3 * edgeLength / dualLength. A single
    // uniform material matches the mean dual length, not every hinge exactly.
    double dualSum = 0;
    unsigned count = 0;
    for (const auto& item : edges)
        if (item.second.faces == 2)
        {
            const float length = (gVertices[item.first.first] - gVertices[item.first.second]).magnitude();
            dualSum += 2.f * item.second.area / (3.f * length);
            ++count;
        }
    return count ? float(1.e-3 * dualSum / count) / (thickness * thickness * thickness) : 0.f;
}

PxVec3 target(unsigned anchor, double time)
{
    const PxVec3& rest = gVertices[gAnchors[anchor]];
    const PxVec3 root(0, rest.y, 0);
    const float angle = float(PxClamp(time, 0.0, 10.0)) * PxPi / 3.f * gSigns[anchor];
    return root + PxQuat(angle, PxVec3(0, 1, 0)).rotate(rest - root);
}

void upload(PxDeformableSurfaceDataFlags flags)
{
    PxDeformableSurfaceExt::copyToDevice(*gCloth, flags, PxU32(gVertices.size()),
        gPositions, gVelocities, gRest);
}

void download()
{
    PxScopedCudaLock lock(*gCuda);
    PxCudaContext* ctx = gCuda->getCudaContext();
    const size_t bytes = gVertices.size() * sizeof(PxVec4);
    ctx->memcpyDtoH(gPositions, reinterpret_cast<CUdeviceptr>(gCloth->getPositionInvMassBufferD()), bytes);
    ctx->memcpyDtoH(gVelocities, reinterpret_cast<CUdeviceptr>(gCloth->getVelocityBufferD()), bytes);
}

void reset()
{
    gFrame = 0;
    for (unsigned i = 0; i < gVertices.size(); ++i)
    {
        gPositions[i] = PxVec4(gVertices[i], 1.f / gMasses[i]);
        gVelocities[i] = PxVec4(0.f);
        gRest[i] = PxVec4(gVertices[i], 0.f);
    }
    for (unsigned i : gAnchors)
        gPositions[i].w = 0.f;
    upload(PxDeformableSurfaceDataFlag::eALL);
    gCloth->setWakeCounter(0.4f);
}

bool init()
{
    gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrors);
    if (!gFoundation) return false;
    PxCudaContextManagerDesc cudaDesc;
    gCuda = PxCreateCudaContextManager(*gFoundation, cudaDesc, NULL);
    if (!gCuda || !gCuda->contextIsValid())
    {
        std::fprintf(stderr, "A working NVIDIA CUDA device is required.\n");
        return false;
    }
    PxTolerancesScale scale;
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, scale);
    if (!gPhysics) return false;
    gExtensions = PxInitExtensions(*gPhysics, NULL);
    if (!gExtensions) return false;
    PxSceneDesc desc(scale);
    desc.gravity = PxVec3(0.f);
    desc.cudaContextManager = gCuda;
    desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS | PxSceneFlag::eENABLE_PCM;
    desc.broadPhaseType = PxBroadPhaseType::eGPU;
    desc.solverType = PxSolverType::eTGS;
    gDispatcher = PxDefaultCpuDispatcherCreate(2);
    desc.cpuDispatcher = gDispatcher;
    desc.filterShader = PxDefaultSimulationFilterShader;
    gScene = gPhysics->createScene(desc);
    if (!gScene) return false;

    PxCookingParams cooking(scale);
    cooking.buildGPUData = true;
    // Preserve source vertex indices: no welding or vertex cleanup/reordering.
    cooking.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH) |
        PxMeshPreprocessingFlag::eFORCE_32BIT_INDICES;
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(gVertices.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = gVertices.data();
    meshDesc.triangles.count = PxU32(gTriangles.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = gTriangles.data();
    gMesh = PxCreateTriangleMesh(cooking, meshDesc, gPhysics->getPhysicsInsertionCallback());
    if (!gMesh || gMesh->getNbVertices() != gVertices.size()) return false;
    for (unsigned i = 0; i < gVertices.size(); ++i)
        if ((gMesh->getVertices()[i] - gVertices[i]).magnitude() > 1.e-7f)
            return false;

    // Effective membrane stiffness ~1000 N/m; parameters are a PhysX material
    // approximation, not a conversion of Newton's global-XPBD constitutive law.
    const float bending = bendingMaterial(0.001f);
    gMaterial = gPhysics->createDeformableSurfaceMaterial(2.5e6f, 0.25f, 0.2f, 0.001f, bending);
    if (!gMaterial) return false;
    gMaterial->setElasticityDamping(600.f);
    gMaterial->setBendingDamping(600.f);
    gCloth = gPhysics->createDeformableSurface(*gCuda);
    if (!gCloth) return false;
    PxShape* shape = gPhysics->createShape(PxTriangleMeshGeometry(gMesh), &gMaterial, 1, true);
    if (!shape) return false;
    // Collision shell thickness is independent of the elastic material thickness.
    shape->setRestOffset(0.002f);
    shape->setContactOffset(0.0035f);
    const bool attached = gCloth->attachShape(*shape);
    shape->release();
    if (!attached) return false;
    gCloth->setSolverIterationCounts(gIterations, 1);
    gCloth->setDeformableSurfaceFlag(PxDeformableSurfaceFlag::eENABLE_SELF_COLLISION_DAT, gDat && gSelfContact);
    gCloth->setLinearDamping(0.f);
    gCloth->setSleepThreshold(0.f);
    gCloth->setSettlingThreshold(0.f);
    gCloth->setSelfCollisionFilterDistance(0.005f);
    gCloth->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !gSelfContact);
    gCloth->setNbCollisionPairUpdatesPerTimestep(1);
    gCloth->setNbCollisionSubsteps(1);
    gScene->addActor(*gCloth);
    PxDeformableSurfaceExt::allocateAndInitializeHostMirror(*gCloth, gVertices.data(), NULL,
        gVertices.data(), 0.3f, PxTransform(PxIdentity), gCuda, gPositions, gVelocities, gRest);
    if (!gPositions || !gVelocities || !gRest) return false;
    reset();
    std::printf("PhysX FEM/TGS: %u substeps, %u iterations, self-contact %s, DAT %s\n",
        gSubsteps, gIterations, gSelfContact ? "on" : "off", gDat && gSelfContact ? "on" : "off");
    return !gErrors.failed;
}

bool validate()
{
    float maxPosition = 0.f, maxError = 0.f;
    for (unsigned i = 0; i < gVertices.size(); ++i)
    {
        if (!gPositions[i].isFinite() || !gVelocities[i].isFinite()) return false;
        const PxVec3 p = gPositions[i].getXYZ();
        maxPosition = PxMax(maxPosition, PxMax(PxAbs(p.x), PxMax(PxAbs(p.y), PxAbs(p.z))));
    }
    const double time = double(gFrame) / 60.0 - 1.0 / 600.0;
    for (unsigned i = 0; i < gAnchors.size(); ++i)
        maxError = PxMax(maxError, (gPositions[gAnchors[i]].getXYZ() - target(i, time)).magnitude());
    const bool valid = maxPosition < 2.f && maxError < 2.e-5f && !gErrors.failed;
    if (gFrame % 60 == 0 || !valid)
        std::printf("frame=%u time=%.3f max_abs_position=%.6f anchor_error=%.9f %s\n",
            gFrame, double(gFrame) / 60.0, maxPosition, maxError, valid ? "PASS" : "FAIL");
    return valid;
}

void step()
{
    const float dt = gFrameDt / float(gSubsteps);
    for (unsigned s = 0; s < gSubsteps; ++s)
    {
        // Endpoint lag and opposing Y-axis rotations match the reference.
        const double time = (double(gFrame) + double(s + 1) / gSubsteps) / 60.0 - 1.0 / 600.0;
        for (unsigned a = 0; a < gAnchors.size(); ++a)
        {
            const unsigned i = gAnchors[a];
            // Zero inverse mass makes these prescribed vertices. Feed velocities
            // so PhysX advances them across its TGS slices toward the target.
            gVelocities[i] = PxVec4((target(a, time) - gPositions[i].getXYZ()) / dt, 0.f);
        }
        // Host buffers retain the previous fetched state for all free vertices.
        upload(PxDeformableSurfaceDataFlag::eVELOCITY);
        gCloth->setWakeCounter(0.4f);
        gScene->simulate(dt);
        gScene->fetchResults(true);
        download();
        if (gErrors.failed)
        {
            gValid = false;
            gPaused = true;
            return;
        }
    }
    ++gFrame;
    gValid = validate();
    if (gDump && (std::fwrite(gPositions, sizeof(PxVec4), gVertices.size(), gDump) != gVertices.size() ||
                  std::fwrite(gVelocities, sizeof(PxVec4), gVertices.size(), gDump) != gVertices.size()))
        gValid = false;
    if (!gValid) gPaused = true;
}

void cleanup()
{
    PX_RELEASE(gCloth);
    if (gCuda)
    {
        PX_EXT_PINNED_MEMORY_FREE(*gCuda, gPositions);
        PX_EXT_PINNED_MEMORY_FREE(*gCuda, gVelocities);
        PX_EXT_PINNED_MEMORY_FREE(*gCuda, gRest);
    }
    PX_RELEASE(gMesh);
    PX_RELEASE(gMaterial);
    PX_RELEASE(gScene);
    PX_RELEASE(gDispatcher);
    if (gExtensions) PxCloseExtensions();
    PX_RELEASE(gPhysics);
    PX_RELEASE(gCuda);
    PX_RELEASE(gFoundation);
}

void keyboard(unsigned char key, const PxTransform&)
{
    if (key == ' ') gPaused = !gPaused;
    if (key == 'n' || key == 'N') gWire = !gWire;
    if (key == 'r' || key == 'R')
    {
        reset();
        gValid = true;
        gPaused = false;
    }
}

void render()
{
    if (!gPaused && gValid && gFrame < gFrames) step();
    Snippets::startRender(gCamera, 0.01f, 100.f);
    // The shared camera uses Y-up; display Newton's Z-up coordinates unchanged
    // in the simulation, with a display-only rotation about the viewing axis.
    glPushMatrix();
    glRotatef(-90.f, 1.f, 0.f, 0.f);
    Snippets::renderMesh(PxU32(gVertices.size()), gPositions, PxU32(gTriangles.size() / 3),
        gTriangles.data(), false, PxVec3(0.3f, 0.65f, 0.9f), NULL, false, true);
    Snippets::renderMesh(PxU32(gVertices.size()), gPositions, PxU32(gTriangles.size() / 3),
        gTriangles.data(), false, PxVec3(0.9f, 0.6f, 0.3f), NULL, true, true);
    glDisable(GL_LIGHTING);
    if (gWire)
    {
        glColor3f(0.08f, 0.12f, 0.18f);
        glBegin(GL_LINES);
        for (unsigned t = 0; t < gTriangles.size(); t += 3)
            for (unsigned e = 0; e < 3; ++e)
            {
                glVertex3fv(&gPositions[gTriangles[t + e]].x);
                glVertex3fv(&gPositions[gTriangles[t + (e + 1) % 3]].x);
            }
        glEnd();
    }
    glPointSize(5.f);
    glBegin(GL_POINTS);
    for (unsigned a = 0; a < gAnchors.size(); ++a)
    {
        glColor3f(gSigns[a] > 0 ? 1.f : 0.2f, gSigns[a] > 0 ? 0.3f : 1.f, 0.2f);
        glVertex3fv(&gPositions[gAnchors[a]].x);
    }
    glEnd();
    glEnable(GL_LIGHTING);
    glPopMatrix();
    char title[200];
    std::snprintf(title, sizeof(title), "PhysX Cloth Twist [%s] | %.2f / %.2f s | %s | Space: pause  R: restart  N: wire",
        gDat && gSelfContact ? "DAT" : "native",
        double(gFrame) / 60.0, double(gFrames) / 60.0,
        !gValid ? "VALIDATION FAILED" : gFrame >= gFrames ? "FINISHED" : gPaused ? "PAUSED" : "RUNNING");
    glutSetWindowTitle(title);
    Snippets::finishRender();
}

bool positiveUnsigned(const char* text, unsigned& value)
{
    char* end = NULL;
    const long parsed = std::strtol(text, &end, 10);
    if (!*text || *end || parsed <= 0 || parsed > 1000000) return false;
    value = unsigned(parsed);
    return true;
}
}

int snippetMain(int argc, const char* const* argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--headless")) gHeadless = true;
        else if (!std::strcmp(argv[i], "--benchmark")) { gBenchmark = true; gHeadless = true; }
        else if (!std::strcmp(argv[i], "--paused")) gPaused = true;
        else if (!std::strcmp(argv[i], "--no-dat")) gDat = false;
        else if (!std::strcmp(argv[i], "--no-self-contact")) gSelfContact = false;
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) gDumpPath = argv[++i];
        else if (!std::strcmp(argv[i], "--mesh") && i + 1 < argc) gMeshPath = argv[++i];
        else if ((!std::strcmp(argv[i], "--frames") || !std::strcmp(argv[i], "--substeps") ||
                  !std::strcmp(argv[i], "--iterations")) && i + 1 < argc)
        {
            unsigned& value = !std::strcmp(argv[i], "--frames") ? gFrames :
                !std::strcmp(argv[i], "--substeps") ? gSubsteps : gIterations;
            if (!positiveUnsigned(argv[++i], value)) return 1;
        }
        else
        {
            std::printf("Usage: %s --mesh square_cloth.mesh [--headless] [--paused] [--frames 600]\n"
                "       [--substeps 10] [--iterations 12] [--no-self-contact] [--no-dat] [--dump PATH] [--benchmark]\n", argv[0]);
            return !std::strcmp(argv[i], "--help") ? 0 : 1;
        }
    }
    if (gIterations > 255 || !loadMesh()) return 1;
    if (!init())
    {
        cleanup();
        return 1;
    }
    if (gDumpPath)
    {
        if (!gHeadless || !(gDump = std::fopen(gDumpPath, "wb")))
        {
            std::fprintf(stderr, "--dump requires headless mode and a writable file.\n");
            cleanup();
            return 1;
        }
        const unsigned header[3] = {unsigned(gVertices.size()), unsigned(gTriangles.size()/3), gFrames};
        std::fwrite("PXCDAT01", 1, 8, gDump);
        std::fwrite(header, sizeof(unsigned), 3, gDump);
        std::fwrite(gVertices.data(), sizeof(PxVec3), gVertices.size(), gDump);
        std::fwrite(gTriangles.data(), sizeof(unsigned), gTriangles.size(), gDump);
    }
    if (gHeadless)
    {
        std::vector<double> timings;
        while (gFrame < gFrames && gValid)
        {
            const auto start = std::chrono::steady_clock::now();
            step();
            const auto end = std::chrono::steady_clock::now();
            if (gBenchmark && gFrame > 10 && gValid)
                timings.push_back(std::chrono::duration<double, std::milli>(end-start).count());
        }
        if (gBenchmark && !timings.empty())
        {
            double total = 0;
            for (double value : timings) total += value;
            std::sort(timings.begin(), timings.end());
            std::printf("BENCHMARK frames=%zu mean_ms=%.6f median_ms=%.6f p95_ms=%.6f\n",
                timings.size(), total/timings.size(), timings[timings.size()/2],
                timings[(timings.size()-1)*95/100]);
        }
    }
    else
    {
        gCamera = new Snippets::Camera(PxVec3(2.25f, 0, 0), PxVec3(-1, 0, 0));
        gCamera->setSpeed(0.05f);
        Snippets::setupDefault("PhysX Cloth Twist", gCamera, keyboard, render, NULL);
        glutMainLoop();
        delete gCamera;
    }
    if (gDump && std::fclose(gDump) != 0) gValid = false;
    cleanup();
    return gValid && !gErrors.failed ? 0 : 1;
}
