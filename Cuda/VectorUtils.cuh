/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

// Cuda
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

// Project
#include "CudaDataTypes.h"

// ________________________________________________________________________________
__device__ __INLINE__ void saturateVector( float4& v )
{
	v.x = (v.x<0.f) ? 0.f : v.x;
	v.y = (v.y<0.f) ? 0.f : v.y; 
	v.z = (v.z<0.f) ? 0.f : v.z;
	v.w = (v.w<0.f) ? 0.f : v.w;

	v.x = (v.x>1.f) ? 1.f : v.x;
	v.y = (v.y>1.f) ? 1.f : v.y; 
	v.z = (v.z>1.f) ? 1.f : v.z;
	v.w = (v.w>1.f) ? 1.f : v.w;
}

// ________________________________________________________________________________
__device__ __INLINE__ Vertex crossProduct( const Vertex& b, const Vertex& c )
{
	Vertex a;
	a.x = b.y*c.z - b.z*c.y;
	a.y = b.z*c.x - b.x*c.z;
	a.z = b.x*c.y - b.y*c.x;
	return a;
}


/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
__device__ __INLINE__ void vectorReflection( Vertex& r, const Vertex& i, const Vertex& n )
{
	r = i-2.f*dot(i,n)*n;
}

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
__device__ __INLINE__ void vectorRefraction( 
	Vertex&      refracted, 
	const Vertex incident, 
	const float  n1, 
	const Vertex normal, 
	const float  n2 )
{
	refracted = incident;
   if(n2!=0.f)
   {
      float eta = n1/n2;
      float c1 = -dot(incident,normal);
      float cs2 = 1.f-eta*eta*(1.f-c1*c1);
      if(cs2>=0.f) 
      {
         refracted = eta*incident + (eta*c1-sqrt(cs2))*normal;
      }
   }
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
__device__ __INLINE__ void vectorRotation( Vertex& v, const Vertex& rotationCenter, const Vertex& angles )
{ 
	Vertex cosAngles, sinAngles;
	
   cosAngles.x = cosf(angles.x);
	cosAngles.y = cosf(angles.y);
	cosAngles.z = cosf(angles.z);
	
   sinAngles.x = sinf(angles.x);
	sinAngles.y = sinf(angles.y);
	sinAngles.z = sinf(angles.z);

   // Rotate Center
   Vertex vector;
   vector.x = v.x - rotationCenter.x;
   vector.y = v.y - rotationCenter.y;
   vector.z = v.z - rotationCenter.z;
   Vertex result = vector; 

   /* X axis */ 
   result.y = vector.y*cosAngles.x - vector.z*sinAngles.x; 
   result.z = vector.y*sinAngles.x + vector.z*cosAngles.x; 
   vector = result; 
   result = vector; 

   /* Y axis */ 
   result.z = vector.z*cosAngles.y - vector.x*sinAngles.y; 
   result.x = vector.z*sinAngles.y + vector.x*cosAngles.y; 
   vector = result; 
   result = vector; 

   /* Z axis */ 
   result.x = vector.x*cosAngles.z - vector.y*sinAngles.z; 
   result.y = vector.x*sinAngles.z + vector.y*cosAngles.z; 

   v.x = result.x + rotationCenter.x; 
   v.y = result.y + rotationCenter.y; 
   v.z = result.z + rotationCenter.z;
}

