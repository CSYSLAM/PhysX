// Experimental Planar-DAT storage for native FEM/TGS surface self-collision.
#ifndef PXG_CLOTH_DAT_H
#define PXG_CLOTH_DAT_H
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxBounds3.h"
#include <vector_types.h>
namespace physx
{
struct PxgClothDatPair
{
    uint4 ids;
    float4 normalGap;
    float4 distances;
    PxU32 kind;
};
struct PxgClothDat
{
    float4* base;
    float4* accepted;
    float* fractions;
    uint2* edges;
    PxBounds3* bounds; // triangles followed by unique edges
    PxgClothDatPair* pairs;
    PxU32* counters; // candidate count, sticky status (1 overflow, 2 separation, 4 finite)
    PxU32 nbEdges;
    PxU32 capacity;
};
}
#endif
