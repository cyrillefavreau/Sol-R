/* 
* Protein Visualizer
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#pragma once

// Cuda
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

// ________________________________________________________________________________
__device__ inline void saturateVector( float4& v )
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
__device__ inline float3 crossProduct( const float3& b, const float3& c )
{
	float3 a;
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
__device__ inline void vectorReflection( float3& r, const float3& i, const float3& n )
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
__device__ inline void vectorRefraction( 
	float3&      refracted, 
	const float3 incident, 
	const float  n1, 
	const float3 normal, 
	const float  n2 )
{
	refracted = incident;
	if(n1!=n2 && n2!=0.f) 
	{
		float r = n1/n2;
		float cosI = dot( incident, normal );
		float cosT2 = 1.f - r*r*(1.f - cosI*cosI);
		refracted = r*incident + (r*cosI-sqrt( fabs(cosT2) ))*normal;
	}
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
__device__ inline void vectorRotation( float3& vector, const float3 center, const float3 angles )
{ 
	float3 result = vector-center; 
	/* X axis */ 
	result.y = vector.y*cos(angles.x) - vector.z*sin(angles.x); 
	result.z = vector.y*sin(angles.x) + vector.z*cos(angles.x); 
	vector = result+center; 
	result = vector-center; 
	/* Y axis */ 
	result.z = vector.z*cos(angles.y) - vector.x*sin(angles.y); 
	result.x = vector.z*sin(angles.y) + vector.x*cos(angles.y); 
	vector = result+center; 
}

