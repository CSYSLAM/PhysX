// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Adapted from Newton solvers/xpbd/fem_contacts.py (Planar-DAT).
// CUDA integration into PhysX native FEM/TGS. This prototype uses exhaustive
// AABB candidate enumeration; it does not inherit PhysX contact filtering.
#include "PxDeformableSurfaceFlag.h"

namespace clothDat
{
typedef physx::PxVec3T<double> D3;
__device__ inline D3 d3(float4 p) { return D3(p.x, p.y, p.z); }
__device__ inline bool enabled(const PxgFEMCloth& f)
{
    return (f.mSurfaceFlags & PxDeformableSurfaceFlag::eENABLE_SELF_COLLISION_DAT) && f.mDat.base;
}
__device__ inline D3 segment(D3 p, D3 a, D3 b)
{
    const D3 u = b-a;
    const double l = u.magnitudeSquared();
    return a + u * (l > 0 ? PxClamp((p-a).dot(u)/l, 0.0, 1.0) : 0.0);
}
__device__ inline D3 triangle(D3 p, D3 a, D3 b, D3 c)
{
    D3 q = segment(p,a,b), r = segment(p,b,c);
    if ((r-p).magnitudeSquared() < (q-p).magnitudeSquared()) q = r;
    r = segment(p,c,a);
    if ((r-p).magnitudeSquared() < (q-p).magnitudeSquared()) q = r;
    const D3 n = (b-a).cross(c-a);
    const double n2 = n.magnitudeSquared();
    if (n2 > 0)
    {
        r = p - n*((p-a).dot(n)/n2);
        if (n.dot((b-a).cross(r-a)) >= 0 && n.dot((c-b).cross(r-b)) >= 0 && n.dot((a-c).cross(r-c)) >= 0)
            q = r;
    }
    return q;
}
__device__ inline void edges(D3 a, D3 b, D3 c, D3 d, D3& p, D3& q)
{
    p = a; q = segment(a,c,d);
    D3 x = b, y = segment(b,c,d);
    if ((x-y).magnitudeSquared() < (p-q).magnitudeSquared()) { p=x; q=y; }
    x = segment(c,a,b); y = c;
    if ((x-y).magnitudeSquared() < (p-q).magnitudeSquared()) { p=x; q=y; }
    x = segment(d,a,b); y = d;
    if ((x-y).magnitudeSquared() < (p-q).magnitudeSquared()) { p=x; q=y; }
    const D3 u=b-a, v=d-c, r=c-a, n=u.cross(v);
    const double n2=n.magnitudeSquared();
    if (n2 > 0)
    {
        const double s=r.cross(v).dot(n)/n2, t=r.cross(u).dot(n)/n2;
        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) { p=a+s*u; q=c+t*v; }
    }
}
__device__ inline bool overlap(PxBounds3 a, PxBounds3 b, float margin)
{
    return a.minimum.x <= b.maximum.x+margin && b.minimum.x <= a.maximum.x+margin &&
           a.minimum.y <= b.maximum.y+margin && b.minimum.y <= a.maximum.y+margin &&
           a.minimum.z <= b.maximum.z+margin && b.minimum.z <= a.maximum.z+margin;
}
__device__ inline void record(PxgClothDat& d, uint4 ids, unsigned kind, D3 a, D3 b)
{
    const unsigned slot = atomicAdd(d.counters, 1u);
    if (slot >= d.capacity) { atomicOr(d.counters+1, 1u); return; }
    const double length = (a-b).magnitude();
    if (length < 1.e-12) { atomicOr(d.counters+1, 2u); return; }
    const D3 n=(a-b)/length;
    double projections[4], lo=1.e300, hi=-1.e300;
    const unsigned v[4]={ids.x,ids.y,ids.z,ids.w};
    for (unsigned j=0; j<4; ++j)
    {
        projections[j]=n.dot(d3(d.base[v[j]])-b);
        if (j==0 || (kind==1 && j==1)) lo=PxMin(lo,projections[j]);
        else hi=PxMax(hi,projections[j]);
    }
    if (lo <= hi) { atomicOr(d.counters+1, 2u); return; }
    const double center=(lo+hi)*0.5;
    PxgClothDatPair pair;
    pair.ids=ids; pair.kind=kind;
    pair.normalGap=make_float4(float(n.x),float(n.y),float(n.z),float(lo-hi));
    pair.distances=make_float4(float(projections[0]-center),float(projections[1]-center),
                              float(projections[2]-center),float(projections[3]-center));
    d.pairs[slot]=pair;
}
}

// Common signature keeps orchestration and stream ordering explicit.
extern "C" __global__ void cloth_datBegin(PxgFEMCloth* cloths, const PxU32* active, float)
{
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!clothDat::enabled(f)) return;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i==0) f.mDat.counters[0]=0; // error status is sticky until actor recreation
    if (i<f.mNbVerts)
    {
        f.mDat.base[i]=f.mDat.accepted[i]=f.mPosition_InvMass[i];
        f.mDat.fractions[i]=1.f;
    }
}
extern "C" __global__ void cloth_datBounds(PxgFEMCloth* cloths, const PxU32* active, float)
{
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!clothDat::enabled(f)) return;
    for (unsigned i=blockIdx.x*blockDim.x+threadIdx.x; i<f.mNbTriangles+f.mDat.nbEdges; i+=gridDim.x*blockDim.x)
    {
        PxBounds3 bound=PxBounds3::empty();
        if (i<f.mNbTriangles)
        {
            const uint4 t=f.mTriangleVertexIndices[i];
            bound.include(PxLoad3(f.mDat.base[t.x])); bound.include(PxLoad3(f.mDat.base[t.y])); bound.include(PxLoad3(f.mDat.base[t.z]));
        }
        else
        {
            const uint2 e=f.mDat.edges[i-f.mNbTriangles];
            bound.include(PxLoad3(f.mDat.base[e.x])); bound.include(PxLoad3(f.mDat.base[e.y]));
        }
        f.mDat.bounds[i]=bound;
    }
}
extern "C" __global__ void cloth_datDetectVTReference(PxgFEMCloth* cloths, const PxU32* active, float)
{
    using namespace clothDat;
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!enabled(f)) return;
    PxgClothDat& d=f.mDat;
    const float margin=f.mOriginalContactOffset;
    const unsigned long long total=static_cast<unsigned long long>(f.mNbVerts)*f.mNbTriangles;
    for (unsigned long long k=blockIdx.x*blockDim.x+threadIdx.x; k<total; k+=gridDim.x*blockDim.x)
    {
        const unsigned i=unsigned(k/f.mNbTriangles), t=unsigned(k%f.mNbTriangles);
        const uint4 face=f.mTriangleVertexIndices[t];
        if (i==face.x || i==face.y || i==face.z) continue;
        const PxVec3 point=PxLoad3(d.base[i]);
        if (!overlap(PxBounds3(point,point),d.bounds[t],margin+2.e-6f*PxMax(1.f,point.magnitude()))) continue;
        const D3 p=d3(d.base[i]);
        const D3 q=triangle(p,d3(d.base[face.x]),d3(d.base[face.y]),d3(d.base[face.z]));
        if ((p-q).magnitudeSquared() <= double(margin)*margin)
            record(d,make_uint4(i,face.x,face.y,face.z),0,p,q);
    }
}
extern "C" __global__ void cloth_datDetectEEReference(PxgFEMCloth* cloths, const PxU32* active, float)
{
    using namespace clothDat;
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!enabled(f)) return;
    PxgClothDat& d=f.mDat;
    const float margin=f.mOriginalContactOffset;
    const unsigned long long total=static_cast<unsigned long long>(d.nbEdges)*d.nbEdges;
    for (unsigned long long k=blockIdx.x*blockDim.x+threadIdx.x; k<total; k+=gridDim.x*blockDim.x)
    {
        const unsigned i=unsigned(k/d.nbEdges), j=unsigned(k%d.nbEdges);
        if (j<=i) continue;
        const uint2 a=d.edges[i], b=d.edges[j];
        if (a.x==b.x || a.x==b.y || a.y==b.x || a.y==b.y) continue;
        const PxBounds3 ba=d.bounds[f.mNbTriangles+i], bb=d.bounds[f.mNbTriangles+j];
        const float reserve=2.e-6f*PxMax(1.f,PxMax(ba.minimum.magnitude(),ba.maximum.magnitude()));
        if (!overlap(ba,bb,margin+reserve)) continue;
        D3 p,q;
        edges(d3(d.base[a.x]),d3(d.base[a.y]),d3(d.base[b.x]),d3(d.base[b.y]),p,q);
        if ((p-q).magnitudeSquared() <= double(margin)*margin)
            record(d,make_uint4(a.x,a.y,b.x,b.y),1,p,q);
    }
}
extern "C" __global__ void cloth_datDetectVT(PxgFEMCloth* cloths, const PxU32* active, float)
{
    using namespace clothDat;
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!enabled(f)) return;
    PxgClothDat& d=f.mDat;
    const float margin=f.mOriginalContactOffset;
    for (unsigned i=blockIdx.x; i<f.mNbVerts; i+=gridDim.x)
    {
        const PxVec3 point=PxLoad3(d.base[i]);
        const PxBounds3 query(point,point);
        const float radius=margin+2.e-6f*PxMax(1.f,point.magnitude());
        const D3 p=d3(d.base[i]);
        for (unsigned t=threadIdx.x; t<f.mNbTriangles; t+=blockDim.x)
        {
            const uint4 face=f.mTriangleVertexIndices[t];
            if (i==face.x || i==face.y || i==face.z || !overlap(query,d.bounds[t],radius)) continue;
            const D3 q=triangle(p,d3(d.base[face.x]),d3(d.base[face.y]),d3(d.base[face.z]));
            if ((p-q).magnitudeSquared()<=double(margin)*margin)
                record(d,make_uint4(i,face.x,face.y,face.z),0,p,q);
        }
    }
}
extern "C" __global__ void cloth_datDetectEE(PxgFEMCloth* cloths, const PxU32* active, float)
{
    using namespace clothDat;
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!enabled(f)) return;
    PxgClothDat& d=f.mDat;
    const float margin=f.mOriginalContactOffset;
    for (unsigned i=blockIdx.x; i<d.nbEdges; i+=gridDim.x)
    {
        const uint2 a=d.edges[i];
        const PxBounds3 ba=d.bounds[f.mNbTriangles+i];
        const float radius=margin+2.e-6f*PxMax(1.f,PxMax(ba.minimum.magnitude(),ba.maximum.magnitude()));
        const D3 p0=d3(d.base[a.x]), p1=d3(d.base[a.y]);
        for (unsigned j=i+1+threadIdx.x; j<d.nbEdges; j+=blockDim.x)
        {
            const uint2 b=d.edges[j];
            if (a.x==b.x || a.x==b.y || a.y==b.x || a.y==b.y || !overlap(ba,d.bounds[f.mNbTriangles+j],radius)) continue;
            D3 p,q;
            edges(p0,p1,d3(d.base[b.x]),d3(d.base[b.y]),p,q);
            if ((p-q).magnitudeSquared()<=double(margin)*margin)
                record(d,make_uint4(a.x,a.y,b.x,b.y),1,p,q);
        }
    }
}
extern "C" __global__ void cloth_datTruncate(PxgFEMCloth* cloths, const PxU32* active, float)
{
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!clothDat::enabled(f)) return;
    PxgClothDat& d=f.mDat;
    // Validate unpaired vertices too; the next kernel observes the final sticky status.
    for (unsigned i=blockIdx.x*blockDim.x+threadIdx.x; i<f.mNbVerts; i+=gridDim.x*blockDim.x)
        if (!PxLoad3(f.mPosition_InvMass[i]).isFinite()) atomicOr(d.counters+1,4u);
    // One atomic read per warp avoids serializing every lane on the sticky flag.
    const unsigned status = (threadIdx.x & 31) == 0 ? atomicAdd(d.counters+1,0u) : 0u;
    if (__shfl_sync(0xffffffffu,status,0)) return;
    for (unsigned k=blockIdx.x*blockDim.x+threadIdx.x; k<PxMin(d.counters[0],d.capacity); k+=gridDim.x*blockDim.x)
    {
        const PxgClothDatPair pair=d.pairs[k];
        const unsigned ids[4]={pair.ids.x,pair.ids.y,pair.ids.z,pair.ids.w};
        const float dist[4]={pair.distances.x,pair.distances.y,pair.distances.z,pair.distances.w};
        const PxVec3 n=PxLoad3(pair.normalGap);
        float da=0,db=0,lo=-0.5f*pair.normalGap.w,hi=0.5f*pair.normalGap.w,current[4];
        for (unsigned j=0;j<4;++j)
        {
            const unsigned i=ids[j];
            const PxVec3 accepted=PxLoad3(d.accepted[i]);
            const float approach=n.dot(PxLoad3(f.mPosition_InvMass[i])-accepted);
            current[j]=dist[j]+n.dot(accepted-PxLoad3(d.base[i]));
            if (j==0 || (pair.kind==1 && j==1)) { da=PxMax(da,-approach); hi=PxMin(hi,current[j]); }
            else { db=PxMax(db,approach); lo=PxMax(lo,current[j]); }
        }
        const float ratio=da+db>0 ? PxClamp(db/(da+db),0.05f,0.95f):0.5f;
        const float shift=lo+ratio*(hi-lo);
        for (unsigned j=0;j<4;++j)
        {
            const unsigned i=ids[j];
            const float distance=current[j]-shift;
            const float approach=n.dot(PxLoad3(f.mPosition_InvMass[i])-PxLoad3(d.accepted[i]));
            float t=1.f;
            if (hi<lo) t=0.f;
            else if (distance*approach<0)
            {
                const float reserve=2.e-7f*PxMax(PxLoad3(d.base[i]).magnitude(),0.01f);
                t=PxClamp(0.85f*PxMax(PxAbs(distance)-reserve,0.f)/PxAbs(approach),0.f,1.f);
            }
            // All fractions are nonnegative: unsigned IEEE ordering matches min.
            if (t<1.f) atomicMin(reinterpret_cast<unsigned*>(d.fractions+i),__float_as_uint(t));
        }
    }
}
extern "C" __global__ void cloth_datCommit(PxgFEMCloth* cloths, const PxU32* active, float dt)
{
    using namespace clothDat;
    PxgFEMCloth& f=cloths[active[blockIdx.y]];
    if (!enabled(f)) return;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=f.mNbVerts) return;
    PxgClothDat& d=f.mDat;
    const float4 trial=f.mPosition_InvMass[i];
    PxVec3 result=PxLoad3(d.accepted[i]);
    if (!d.counters[1] && d.fractions[i]>0)
    {
        D3 endpoint=d3(d.accepted[i])+(d3(trial)-d3(d.accepted[i]))*double(d.fractions[i]);
        const D3 delta=endpoint-d3(d.base[i]);
        const double length2=delta.magnitudeSquared(), budget=0.425*double(f.mOriginalContactOffset);
        if (length2>budget*budget) endpoint=d3(d.base[i])+delta*(budget/sqrt(length2));
        result=PxVec3(float(endpoint.x),float(endpoint.y),float(endpoint.z));
    }
    const PxVec3 correction=result-PxLoad3(trial);
    const PxVec3 velocity=d.counters[1] ? PxVec3(0.f) : PxLoad3(f.mVelocity_InvMass[i])+correction/dt;
    const PxVec3 accumulated=d.counters[1] ? PxVec3(0.f) : PxLoad3(f.mAccumulatedDeltaPos[i])+correction;
    f.mPosition_InvMass[i]=make_float4(result.x,result.y,result.z,trial.w);
    f.mVelocity_InvMass[i]=make_float4(velocity.x,velocity.y,velocity.z,trial.w);
    f.mAccumulatedDeltaPos[i]=make_float4(accumulated.x,accumulated.y,accumulated.z,trial.w);
    d.accepted[i]=f.mPosition_InvMass[i];
    d.fractions[i]=1.f; // initialize the next transaction without a reset launch
}
