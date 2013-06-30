/* 
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

// System
#include <iostream>

// Cuda
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
//#include <helper_math.h>

// Project
#include "../Consts.h"
#include "../Logging.h"
#include "VectorUtils.cuh"
#include "GeometryIntersections.cuh"
#include "GeometryShaders.cuh"

// Device resources
Primitive*        d_primitives[MAX_GPU_COUNT];
BoundingBox*      d_boundingBoxes[MAX_GPU_COUNT]; 
int*              d_lamps[MAX_GPU_COUNT];
Material*         d_materials[MAX_GPU_COUNT];
char*             d_textures[MAX_GPU_COUNT];
LightInformation* d_lightInformation[MAX_GPU_COUNT];
float*            d_randoms[MAX_GPU_COUNT];
float4*           d_postProcessingBuffer[MAX_GPU_COUNT];
char*             d_bitmap[MAX_GPU_COUNT];
int2*             d_primitivesXYIds[MAX_GPU_COUNT];
cudaStream_t      d_streams[MAX_GPU_COUNT];
int               d_nbGPUs;

/*
________________________________________________________________________________

Calculate the reflected vector                   

^ Normal to object surface (N)  
Reflection (O_R)  |                              
\ |  Eye (O_E)                    
\| /                             
----------------O--------------- Object surface 
closestIntersection                      

============================================================================== 
colours                                                                                    
------------------------------------------------------------------------------ 
We now have to know the colour of this intersection                                        
Color_from_object will compute the amount of light received by the
intersection float3 and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the intersection (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
__device__ inline float4 launchRay( 
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	LightInformation* lightInformation, const int& lightInformationSize, const int& nbActiveLamps,
	Material*  materials, char* textures,
	float*           randoms,
	const Ray&       ray, 
	const SceneInfo& sceneInfo,
	const PostProcessingInfo& postProcessingInfo,
	float3&          intersection,
	float&           depthOfField,
	int2&            primitiveXYId)
{
	float4 intersectionColor   = {0.f,0.f,0.f,0.f};

	float3 closestIntersection = {0.f,0.f,0.f};
	float3 firstIntersection   = {0.f,0.f,0.f};
	float3 normal              = {0.f,0.f,0.f};
	int    closestPrimitive  = 0;
	bool   carryon           = true;
	Ray    rayOrigin         = ray;
	float  initialRefraction = 1.f;
	int    iteration         = 0;
	primitiveXYId.x = -1;
   int currentMaterialId=-2;

   // TODO
   float  colorContributions[NB_MAX_ITERATIONS];
   float4 colors[NB_MAX_ITERATIONS];
   memset(&colorContributions[0],0,sizeof(float)*NB_MAX_ITERATIONS);
   memset(&colors[0],0,sizeof(float4)*NB_MAX_ITERATIONS);

	float4 recursiveBlinn = { 0.f, 0.f, 0.f, 0.f };

	// Variable declarations
	float  shadowIntensity = 0.f;
	float4 refractionFromColor;
	float3 reflectedTarget;
	float4 colorBox = {0.f,0.f,0.f,0.f};
	bool   back = false;

#if 0
   // Photon energy
   float photonDistance = sceneInfo.viewDistance.x;
   float previousTransparency = 1.f;
#endif // 0

   // Reflected rays
   int reflectedRays=-1;
   Ray reflectedRay;
   float reflectedRatio;

   float4 rBlinn = {0.f,0.f,0.f,0.f};
   int currentMaxIteration = ( sceneInfo.graphicsLevel.x<3 ) ? 1 : sceneInfo.nbRayIterations.x+sceneInfo.pathTracingIteration.x;
   currentMaxIteration = (currentMaxIteration>NB_MAX_ITERATIONS) ? NB_MAX_ITERATIONS : currentMaxIteration;
	while( iteration<currentMaxIteration && carryon /*&& photonDistance>0.f*/ ) 
	{
      float3 areas = {0.f,0.f,0.f};
      // If no intersection with lamps detected. Now compute intersection with Primitives
		if( carryon ) 
		{
			carryon = intersectionWithPrimitives(
				sceneInfo,
				boundingBoxes, nbActiveBoxes,
				primitives, nbActivePrimitives,
				materials, textures,
				rayOrigin,
				iteration,  
				closestPrimitive, closestIntersection, 
				normal, areas, colorBox, back, currentMaterialId);
		}

		if( carryon ) 
		{
         currentMaterialId = primitives[closestPrimitive].materialId.x;

			if ( iteration==0 )
			{
            colors[iteration].x = 0.f;
            colors[iteration].y = 0.f;
            colors[iteration].z = 0.f;
            colors[iteration].w = 0.f;
            colorContributions[iteration]=1.f;

				firstIntersection = closestIntersection;
            primitiveXYId.x = primitives[closestPrimitive].index.x;
			}

#if 0
         // Photon
         photonDistance -= length(closestIntersection-rayOrigin.origin) * (2.f-previousTransparency);
         previousTransparency = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].transparency.x;
#endif // 0

			// Get object color
         colors[iteration] =
            primitiveShader( 
				   sceneInfo, postProcessingInfo,
				   boundingBoxes, nbActiveBoxes, 
			      primitives, nbActivePrimitives, 
               lightInformation, lightInformationSize, nbActiveLamps,
               materials, textures, 
               randoms, rayOrigin.origin, normal, 
               closestPrimitive, closestIntersection, areas, 
			      iteration, refractionFromColor, shadowIntensity, rBlinn );

			// ----------
			// Refraction
			// ----------

			if( materials[primitives[closestPrimitive].materialId.x].transparency.x != 0.f ) 
			{
			   // Replace the normal using the intersection color
			   // r,g,b become x,y,z... What the fuck!!
			   if( materials[primitives[closestPrimitive].materialId.x].textureMapping.z != TEXTURE_NONE) 
			   {
				   normal.x *= (colors[iteration].x-0.5f);
				   normal.y *= (colors[iteration].y-0.5f);
				   normal.z *= (colors[iteration].z-0.5f);
			   }

				// Back of the object? If so, reset refraction to 1.f (air)
				float refraction = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].refraction.x;

				// Actual refraction
				float3 O_E = normalize(rayOrigin.origin - closestIntersection);
				vectorRefraction( rayOrigin.direction, O_E, refraction, normal, initialRefraction );
				reflectedTarget = closestIntersection - rayOrigin.direction;

            colorContributions[iteration] = materials[primitives[closestPrimitive].materialId.x].transparency.x;
               
            // Prepare next ray
				initialRefraction = refraction;

				if( reflectedRays==-1 && materials[primitives[closestPrimitive].materialId.x].reflection.x != 0.f )
            {
					vectorReflection( reflectedRay.direction, O_E, normal );
					float3 rt = closestIntersection - reflectedRay.direction;

               reflectedRay.origin    = closestIntersection + rt*0.00001f;
					reflectedRay.direction = rt;
               reflectedRatio = materials[primitives[closestPrimitive].materialId.x].reflection.x;
					reflectedRays=iteration;
            }
			}
			else
			{
				// ----------
				// Reflection
				// ----------
				if( materials[primitives[closestPrimitive].materialId.x].reflection.x != 0.f ) 
				{
					float3 O_E = rayOrigin.origin - closestIntersection;
					vectorReflection( rayOrigin.direction, O_E, normal );
					reflectedTarget = closestIntersection - rayOrigin.direction;
               colorContributions[iteration] = materials[primitives[closestPrimitive].materialId.x].reflection.x;
				}
				else 
				{
               colorContributions[iteration] = 1.f;
					carryon = false;
				}         
			}

         // Contribute to final color
 			recursiveBlinn += rBlinn;

         rayOrigin.origin    = closestIntersection + reflectedTarget*0.00001f; 
			rayOrigin.direction = reflectedTarget;

			// Noise management
			if( sceneInfo.pathTracingIteration.x != 0 && materials[primitives[closestPrimitive].materialId.x].color.w != 0.f)
			{
				// Randomize view
            float ratio = materials[primitives[closestPrimitive].materialId.x].color.w;
            ratio *= (materials[primitives[closestPrimitive].materialId.x].transparency.x==0.f) ? 1000.f : 1.f;
				int rindex = 3*sceneInfo.misc.y + sceneInfo.pathTracingIteration.x;
				rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
				rayOrigin.direction.x += randoms[rindex  ]*ratio;
				rayOrigin.direction.y += randoms[rindex+1]*ratio;
				rayOrigin.direction.z += randoms[rindex+2]*ratio;
			}
		}
		else
		{
#if 1
         // Background
         float3 normal = {0.f, 1.f, 0.f };
         float3 dir = normalize(rayOrigin.direction-rayOrigin.origin);
         float angle = 2.f*fabs(dot( normal, dir));
         angle = (angle>1.f) ? 1.f: angle;
			colors[iteration] = (1.f-angle)*sceneInfo.backgroundColor;
#else
			colors[iteration] = sceneInfo.backgroundColor;
#endif // 0
			colorContributions[iteration] = 1.f;
		}
		iteration++;
	}

   if( sceneInfo.graphicsLevel.x>=3 && reflectedRays != -1 ) // TODO: Draft mode should only test "sceneInfo.pathTracingIteration.x==iteration"
   {
      float3 areas = {0.f,0.f,0.f};
      // TODO: Dodgy implementation		
      if( intersectionWithPrimitives(
			sceneInfo,
			boundingBoxes, nbActiveBoxes,
			primitives, nbActivePrimitives,
			materials, textures,
			reflectedRay,
			reflectedRays,  
			closestPrimitive, closestIntersection, 
			normal, areas, colorBox, back, currentMaterialId) )
      {
         float4 color = primitiveShader( 
				sceneInfo, postProcessingInfo,
				boundingBoxes, nbActiveBoxes, 
			   primitives, nbActivePrimitives, 
            lightInformation, lightInformationSize, nbActiveLamps, 
            materials, textures, randoms, 
            reflectedRay.origin, normal, closestPrimitive, 
            closestIntersection, areas, 
			   reflectedRays, 
            refractionFromColor, shadowIntensity, rBlinn );

         colors[reflectedRays] += color*reflectedRatio;
      }
   }

   for( int i=iteration-2; i>=0; --i)
   {
      colors[i] = colors[i]*(1.f-colorContributions[i]) + colors[i+1]*colorContributions[i];
   }
   intersectionColor = colors[0];
	intersectionColor += recursiveBlinn;

	intersection = closestIntersection;

   Primitive& primitive=primitives[closestPrimitive];
	float len = length(firstIntersection - ray.origin);
   if( materials[primitive.materialId.x].attributes.z == 0 ) // Wireframe
   {
#if 0
	   // --------------------------------------------------
      // Photon energy
	   // --------------------------------------------------
      intersectionColor *= ( photonDistance>0.f) ? (photonDistance/sceneInfo.viewDistance.x) : 0.f;
#endif // 0

	   // --------------------------------------------------
	   // Fog
	   // --------------------------------------------------
      //intersectionColor += randoms[((int)len + sceneInfo.misc.y)%100];

	   // --------------------------------------------------
	   // Attenation effect
	   // --------------------------------------------------
      float D1 = sceneInfo.viewDistance.x*0.95f;
      if( sceneInfo.misc.z==1 && len>D1)
      {
         float D2 = sceneInfo.viewDistance.x*0.05f;
         float a = len - D1;
         float b = 1.f-(a/D2);
         intersectionColor = intersectionColor*b + sceneInfo.backgroundColor*(1.f-b);
      }
   }
   depthOfField = (len-depthOfField)/sceneInfo.viewDistance.x;

   // Primitive information
   primitiveXYId.y = iteration;

	// Depth of field
   intersectionColor -= colorBox;
	saturateVector( intersectionColor );
	return intersectionColor;
}


/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
__global__ void k_standardRenderer(
   int split_y, int nbGPUs,
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
   Material*    materials,
	char*        textures,
	float*       randoms,
	float3       origin,
	float3       direction,
	float3       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int2*        primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	Ray ray;
	ray.origin = origin;
	ray.direction = direction;

   float3 rotationCenter = {0.f,0.f,0.f};
   bool antialiasingActivated = (sceneInfo.misc.w == 2);
   
	if( sceneInfo.pathTracingIteration.x == 0 )
   {
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
   }
   else
	{
		// Randomize view for natural depth of field
      if( sceneInfo.pathTracingIteration.x >= NB_MAX_ITERATIONS )
      {
		   int rindex = index + sceneInfo.pathTracingIteration.x;
		   rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
		   ray.direction.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		   ray.direction.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		   ray.direction.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
      }
	}

	float dof = postProcessingInfo.param1.x;
	float3 intersection;


   if( sceneInfo.misc.w == 1 ) // Isometric 3D
   {
      ray.direction.x = ray.origin.z*0.001f*(float)(x - (sceneInfo.width.x/2));
	   ray.direction.y = -ray.origin.z*0.001f*(float)(split_y+y - (sceneInfo.height.x/2));
	   ray.origin.x = ray.direction.x;
	   ray.origin.y = ray.direction.y;
   }
   else
   {
      float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
      float2 step;
      step.x=ratio*6400.f/(float)sceneInfo.width.x;
      step.y=6400.f/(float)sceneInfo.height.x;
      ray.direction.x = ray.direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
      ray.direction.y = ray.direction.y + step.y*(float)(split_y+y - (sceneInfo.height.x/2));
   }

	vectorRotation( ray.origin, rotationCenter, angles );
	vectorRotation( ray.direction, rotationCenter, angles );

   // Antialisazing
   float2 AArotatedGrid[4] =
   {
      {  3.f,  5.f },
      {  5.f, -3.f },
      { -3.f, -5.f },
      { -5.f,  3.f }
   };

   if( sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y && sceneInfo.pathTracingIteration.x>0 && sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS ) return;

   float4 color = {0.f,0.f,0.f,0.f};
   if( antialiasingActivated )
   {
      Ray r=ray;
	   for( int I=0; I<4; ++I )
	   {
         r.direction.x = ray.direction.x + AArotatedGrid[I].x;
         r.direction.y = ray.direction.y + AArotatedGrid[I].y;
	      float4 c = launchRay(
		      BoundingBoxes, nbActiveBoxes,
		      primitives, nbActivePrimitives,
            lightInformation, lightInformationSize, nbActiveLamps,
		      materials, textures, 
		      randoms,
		      r, 
		      sceneInfo, postProcessingInfo,
		      intersection,
		      dof,
		      primitiveXYIds[index]);
         color += c;
      }
   }
   color += launchRay(
		BoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
      lightInformation, lightInformationSize, nbActiveLamps,
		materials, textures, 
		randoms,
		ray, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);
   
   if( antialiasingActivated )
   {
      color /= 5.f;
   }

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].w = dof;
	}

   if( sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = color.x;
      postProcessingBuffer[index].y = color.y;
      postProcessingBuffer[index].z = color.z;
   }
   else
   {
      postProcessingBuffer[index].x += color.x;
      postProcessingBuffer[index].y += color.y;
      postProcessingBuffer[index].z += color.z;
   }
}

/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
__global__ void k_fishEyeRenderer(
   int split_y, int nbGPUs,
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float3       origin,
	float3       direction,
	float3       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int2*        primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	Ray ray;
	ray.origin = origin;
	ray.direction = direction;

	if( sceneInfo.pathTracingIteration.x == 0 )
   {
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
   }
   else
	{
		// Randomize view for natural depth of field
      if( sceneInfo.pathTracingIteration.x >= NB_MAX_ITERATIONS )
      {
		   int rindex = index + sceneInfo.pathTracingIteration.x;
		   rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
		   ray.direction.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		   ray.direction.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		   ray.direction.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
      }
	}

	float dof = postProcessingInfo.param1.x;
	float3 intersection;

   // Normal Y axis
   float2 step;
   step.y=6400.f/(float)sceneInfo.height.x;
   ray.direction.y = ray.direction.y + step.y*(float)(split_y+y - (sceneInfo.height.x/2));

   // 360° X axis
   step.x = 2.f*M_PI/sceneInfo.width.x;
   step.y = 2.f*M_PI/sceneInfo.height.x;

   float3 fishEyeAngles = {0.f,0.f,0.f};
   fishEyeAngles.y = angles.y + step.x*(float)x;
   //fishEyeAngles.x = angles.x + step.y*(float)y;

   vectorRotation( ray.direction, ray.origin, fishEyeAngles );

	//vectorRotation( ray.origin,    rotationCenter, angles );
	//vectorRotation( ray.direction, rotationCenter, angles );
	
   if( sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y && sceneInfo.pathTracingIteration.x>0 && sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS ) return;

   float4 color = {0.f,0.f,0.f,0.f};
   color += launchRay(
		BoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lightInformation, lightInformationSize, nbActiveLamps,
		materials, textures, 
		randoms,
		ray, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);
   
	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].w = dof;
	}

   if( sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = color.x;
      postProcessingBuffer[index].y = color.y;
      postProcessingBuffer[index].z = color.z;
   }
   else
   {
      postProcessingBuffer[index].x += color.x;
      postProcessingBuffer[index].y += color.y;
      postProcessingBuffer[index].z += color.z;
   }
}

/*
________________________________________________________________________________

Anaglyph Renderer
________________________________________________________________________________
*/
__global__ void k_anaglyphRenderer(
	BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float3       origin,
	float3       direction,
	float3       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int2*        primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	float3 rotationCenter = {0.f,0.f,0.f};

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	float3 intersection;
	Ray eyeRay;

   float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
   float2 step;
   step.x=4.f*ratio*6400.f/(float)sceneInfo.width.x;
   step.y=4.f*6400.f/(float)sceneInfo.height.x;

   // Left eye
	eyeRay.origin.x = origin.x + sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 colorLeft = launchRay(
		boundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lightInformation, lightInformationSize, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	// Right eye
	eyeRay.origin.x = origin.x - sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );
	float4 colorRight = launchRay(
		boundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lightInformation, lightInformationSize, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	float r1 = colorLeft.x*0.299f + colorLeft.y*0.587f + colorLeft.z*0.114f;
	float b1 = 0.f;
	float g1 = 0.f;

	float r2 = 0.f;
	float g2 = colorRight.y;
	float b2 = colorRight.z;

	if( sceneInfo.pathTracingIteration.x == 0 ) postProcessingBuffer[index].w = dof;
   if( sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = r1+r2;
      postProcessingBuffer[index].y = g1+g2;
      postProcessingBuffer[index].z = b1+b2;
   }
   else
   {
      postProcessingBuffer[index].x += r1+r2;
      postProcessingBuffer[index].y += g1+g2;
      postProcessingBuffer[index].z += b1+b2;
   }
}

/*
________________________________________________________________________________

3D Vision Renderer
________________________________________________________________________________
*/
__global__ void k_3DVisionRenderer(
	BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive*   primitives,    int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float3       origin,
	float3       direction,
	float3       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int2*        primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	float3 rotationCenter = {0.f,0.f,0.f};

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	float3 intersection;
	int halfWidth  = sceneInfo.width.x/2;

   float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
   float2 step;
   step.x=ratio*6400.f/(float)sceneInfo.width.x;
   step.y=6400.f/(float)sceneInfo.height.x;

   Ray eyeRay;
	if( x<halfWidth ) 
	{
		// Left eye
		eyeRay.origin.x = origin.x + sceneInfo.width3DVision.x;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

		eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2) + halfWidth/2 );
		eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}
	else
	{
		// Right eye
		eyeRay.origin.x = origin.x - sceneInfo.width3DVision.x;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

		eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2) - halfWidth/2);
		eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 color = launchRay(
		boundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lightInformation, lightInformationSize, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	if( sceneInfo.pathTracingIteration.x == 0 ) postProcessingBuffer[index].w = dof;
   if( sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = color.x;
      postProcessingBuffer[index].y = color.y;
      postProcessingBuffer[index].z = color.z;
   }
   else
   {
      postProcessingBuffer[index].x += color.x;
      postProcessingBuffer[index].y += color.y;
      postProcessingBuffer[index].z += color.z;
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Default
________________________________________________________________________________
*/
__global__ void k_default(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
   int2*            primitiveXYIds,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   float4 localColor = postProcessingBuffer[index];
   
   if(sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
      localColor /= (float)(sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Depth of field
________________________________________________________________________________
*/
__global__ void k_depthOfField(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float  depth = PostProcessingInfo.param2.x*postProcessingBuffer[index].w;
	int    wh = sceneInfo.width.x*sceneInfo.height.x;

   float4 localColor = {0.f,0.f,0.f};
	for( int i=0; i<PostProcessingInfo.param3.x; ++i )
	{
		int ix = i%wh;
		int iy = (i+sceneInfo.width.x)%wh;
		int xx = x+depth*randoms[ix]*0.5f;
		int yy = y+depth*randoms[iy]*0.5f;
		if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
		{
			int localIndex = yy*sceneInfo.width.x+xx;
			if( localIndex>=0 && localIndex<wh )
			{
				localColor += postProcessingBuffer[localIndex];
			}
		}
		else
		{
			localColor += postProcessingBuffer[index];
		}
	}
	localColor /= PostProcessingInfo.param3.x;

   if(sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
      localColor /= (float)(sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);

	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
__global__ void k_ambiantOcclusion(
	SceneInfo        sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float occ = 0.f;
	float4 localColor = postProcessingBuffer[index];
	float  depth = localColor.w;

   const int step = 16;
	for( int X=-step; X<step; X+=2 )
	{
		for( int Y=-step; Y<step; Y+=2 )
		{
			int xx = x+X;
			int yy = y+Y;
			if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
			{
				int localIndex = yy*sceneInfo.width.x+xx;
				if( postProcessingBuffer[localIndex].w>=depth)
				{
					occ += 1.f;
				}
			}
			else
				occ += 1.f;
		}
	}
	//occ /= float((2*step)*(2*step));
	occ /= float(step*step);
	occ += 0.3f; // Ambient light
	localColor.x *= occ;
	localColor.y *= occ;
	localColor.z *= occ;

   if(sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
      localColor /= (float)(sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);

	saturateVector( localColor );
	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Cartoon
________________________________________________________________________________
*/
__global__ void k_cartoon(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float4 localColor = postProcessingBuffer[index];

	int r = localColor.x*255/PostProcessingInfo.param3.x;
	int g = localColor.y*255/PostProcessingInfo.param3.x;
	int b = localColor.z*255/PostProcessingInfo.param3.x;

	localColor.x = float(r*PostProcessingInfo.param3.x/255.f);
	localColor.y = float(g*PostProcessingInfo.param3.x/255.f);
	localColor.z = float(b*PostProcessingInfo.param3.x/255.f);

   if(sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
      localColor /= (float)(sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);

	localColor.w = 1.f;
	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

GPU initialization
________________________________________________________________________________
*/
extern "C" void initialize_scene( 
	int width, int height, int nbPrimitives, int nbLamps, int nbMaterials )
{
#if 0
   // Multi GPU initialization
   checkCudaErrors(cudaGetDeviceCount(&d_nbGPUs));
   if(d_nbGPUs > MAX_GPU_COUNT)
   {
	   d_nbGPUs = MAX_GPU_COUNT;
   }
#else
   d_nbGPUs = 1;
#endif
   LOG_INFO(1 ,"CUDA-capable device count: " << d_nbGPUs);

   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
      checkCudaErrors(cudaStreamCreate(&d_streams[d]));

      // Scene resources
	   checkCudaErrors(cudaMalloc( (void**)&d_boundingBoxes[d],      NB_MAX_BOXES*sizeof(BoundingBox)));
	   checkCudaErrors(cudaMalloc( (void**)&d_primitives[d],         NB_MAX_PRIMITIVES*sizeof(Primitive)));
	   checkCudaErrors(cudaMalloc( (void**)&d_lamps[d],              NB_MAX_LAMPS*sizeof(int)));
	   checkCudaErrors(cudaMalloc( (void**)&d_materials[d],          NB_MAX_MATERIALS*sizeof(Material)));
	   checkCudaErrors(cudaMalloc( (void**)&d_lightInformation[d],   NB_MAX_LIGHTINFORMATIONS*sizeof(LightInformation)));
	   checkCudaErrors(cudaMalloc( (void**)&d_randoms[d],            width*height*sizeof(float)));

	   // Rendering canvas
	   checkCudaErrors(cudaMalloc( (void**)&d_postProcessingBuffer[d],  width*height*sizeof(float4)/d_nbGPUs));
	   checkCudaErrors(cudaMalloc( (void**)&d_bitmap[d],                width*height*gColorDepth*sizeof(char)/d_nbGPUs));
	   checkCudaErrors(cudaMalloc( (void**)&d_primitivesXYIds[d],       width*height*sizeof(int2)/d_nbGPUs));
   }
#if 1
	LOG_INFO( 3, "GPU: SceneInfo         : " << sizeof(SceneInfo) );
	LOG_INFO( 3, "GPU: Ray               : " << sizeof(Ray) );
	LOG_INFO( 3, "GPU: PrimitiveType     : " << sizeof(PrimitiveType) );
	LOG_INFO( 3, "GPU: Material          : " << sizeof(Material) );
	LOG_INFO( 3, "GPU: BoundingBox       : " << sizeof(BoundingBox) );
	LOG_INFO( 3, "GPU: Primitive         : " << sizeof(Primitive) );
	LOG_INFO( 3, "GPU: PostProcessingType: " << sizeof(PostProcessingType) );
	LOG_INFO( 3, "GPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) );
	LOG_INFO( 3, "Textures " << NB_MAX_TEXTURES );
#endif // 0
}

/*
________________________________________________________________________________

GPU finalization
________________________________________________________________________________
*/
extern "C" void finalize_scene()
{
   LOG_INFO(1 ,"Releasing device resources");
   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
	   checkCudaErrors(cudaFree( d_boundingBoxes[d] ));
	   checkCudaErrors(cudaFree( d_primitives[d] ));
	   checkCudaErrors(cudaFree( d_lamps[d] ));
	   checkCudaErrors(cudaFree( d_materials[d] ));
	   checkCudaErrors(cudaFree( d_textures[d] ));
	   checkCudaErrors(cudaFree( d_lightInformation[d] ));
	   checkCudaErrors(cudaFree( d_randoms[d] ));
	   checkCudaErrors(cudaFree( d_postProcessingBuffer[d] ));
	   checkCudaErrors(cudaFree( d_bitmap[d] ));
	   checkCudaErrors(cudaFree( d_primitivesXYIds[d] ));
      checkCudaErrors(cudaStreamDestroy(d_streams[d]));
   }
}

/*
________________________________________________________________________________

CPU -> GPU data transfers
________________________________________________________________________________
*/
extern "C" void h2d_scene( 
	BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive*  primitives, int nbPrimitives,
	int* lamps, int nbLamps )
{
   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
	   checkCudaErrors(cudaMemcpyAsync( d_boundingBoxes[d],      boundingBoxes,      nbActiveBoxes*sizeof(BoundingBox), cudaMemcpyHostToDevice, d_streams[d] ));
	   checkCudaErrors(cudaMemcpyAsync( d_primitives[d],         primitives,         nbPrimitives*sizeof(Primitive),    cudaMemcpyHostToDevice, d_streams[d] ));
	   checkCudaErrors(cudaMemcpyAsync( d_lamps[d],              lamps,              nbLamps*sizeof(int),               cudaMemcpyHostToDevice, d_streams[d] ));
   }
}

extern "C" void h2d_materials( 
	Material*  materials, int nbActiveMaterials,
	float*     randoms,   int nbRandoms)
{
   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
	   checkCudaErrors(cudaMemcpyAsync( d_materials[d], materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice, d_streams[d] ));
	   checkCudaErrors(cudaMemcpyAsync( d_randoms[d],   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice, d_streams[d] ));
   }
}

extern "C" void h2d_textures( 
	const int activeTextures, TextureInformation* textureInfos )
{
   for( int d(0); d<d_nbGPUs; ++d )
   {
      int totalSize=0;
      for( int i(0); i<activeTextures; ++i )
      {
         totalSize += textureInfos[i].size.x*textureInfos[i].size.y*textureInfos[i].size.z;
      }
	   checkCudaErrors(cudaFree( d_textures[d] ));
	   checkCudaErrors(cudaMalloc( (void**)&d_textures[d], totalSize*sizeof(char)));

      checkCudaErrors(cudaSetDevice(d));
      for( int i(0); i<activeTextures; ++i )
      {
         if( textureInfos[i].buffer != nullptr )
         {
            int textureSize = textureInfos[i].size.x*textureInfos[i].size.y*textureInfos[i].size.z;
	         checkCudaErrors(cudaMemcpyAsync( d_textures[d]+textureInfos[i].offset, textureInfos[i].buffer, textureSize*sizeof(char), cudaMemcpyHostToDevice, d_streams[d] ));
         }
      }
   }
}

extern "C" void h2d_lightInformation( 
	LightInformation* lightInformation , int lightInformationSize)
{
   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
	   checkCudaErrors(cudaMemcpyAsync( d_lightInformation[d],  lightInformation,  lightInformationSize*sizeof(LightInformation), cudaMemcpyHostToDevice, d_streams[d] ));
   }
}

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
	char* kinectVideo, char* kinectDepth )
{
   for( int d(0); d<d_nbGPUs; ++d )
   {
	   checkCudaErrors(cudaMemcpyAsync( d_textures[d], kinectVideo, gKinectVideoSize*sizeof(char), cudaMemcpyHostToDevice, d_streams[d] ));
	   checkCudaErrors(cudaMemcpyAsync( d_textures[d]+gKinectVideoSize, kinectDepth, gKinectDepthSize*sizeof(char), cudaMemcpyHostToDevice, d_streams[d] ));
   }
}
#endif // USE_KINECT

/*
________________________________________________________________________________

GPU -> CPU data transfers
________________________________________________________________________________
*/
extern "C" void d2h_bitmap( unsigned char* bitmap, int2* primitivesXYIds, const SceneInfo sceneInfo )
{
   int offsetBitmap = sceneInfo.width.x*sceneInfo.height.x*gColorDepth/d_nbGPUs;
   int offsetXYIds  = sceneInfo.width.x*sceneInfo.height.x/d_nbGPUs;
   for( int d(0); d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));
      
      // Synchronize stream
      checkCudaErrors(cudaStreamSynchronize(d_streams[d]));

      // Copy results back to CPU
      checkCudaErrors(cudaMemcpyAsync( bitmap+d*offsetBitmap,         d_bitmap[d],          offsetBitmap*sizeof(char), cudaMemcpyDeviceToHost, d_streams[d] ));
	   checkCudaErrors(cudaMemcpyAsync( primitivesXYIds+d*offsetXYIds, d_primitivesXYIds[d], offsetXYIds*sizeof(int2),   cudaMemcpyDeviceToHost, d_streams[d] ));
   }
}

/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
extern "C" void cudaRender(
	int4 blockSize,
	SceneInfo sceneInfo,
	int4 objects,
	PostProcessingInfo postProcessingInfo,
	float3 origin, 
	float3 direction, 
	float3 angles)
{
   LOG_INFO(3, "GPU Bounding Box: " << sizeof(BoundingBox));
   LOG_INFO(3, "GPU Primitive   : " << sizeof(Primitive));
   LOG_INFO(3, "GPU Material    : " << sizeof(Material));

	LOG_INFO(3, "GPU Boxes              :" << objects.x);
	LOG_INFO(3, "GPU Primitives         :" << objects.y);
	LOG_INFO(3, "GPU Lamps              :" << objects.z);
	LOG_INFO(3, "GPU Light information  :" << objects.w);

	int2 size;
	size.x = static_cast<int>(sceneInfo.width.x);
	size.y = static_cast<int>(sceneInfo.height.x) / d_nbGPUs;

	dim3 grid((size.x+blockSize.x-1)/blockSize.x,(size.y+blockSize.y-1)/blockSize.y,1);
	dim3 blocks( blockSize.x,blockSize.y,blockSize.z );

   for( int d=0; d<d_nbGPUs; ++d )
   {
      checkCudaErrors(cudaSetDevice(d));

	   switch( sceneInfo.renderingType.x ) 
	   {
	   case vtAnaglyph:
		   {
			   k_anaglyphRenderer<<<grid,blocks,0,d_streams[d]>>>(
				   d_boundingBoxes[d], objects.x, 
               d_primitives[d], objects.y,  
               d_lightInformation[d], objects.w, objects.z,
               d_materials[d], d_textures[d], 
				   d_randoms[d], origin, direction, angles, sceneInfo, 
				   postProcessingInfo, d_postProcessingBuffer[d], d_primitivesXYIds[d]);
			   break;
		   }
	   case vt3DVision:
		   {
			   k_3DVisionRenderer<<<grid,blocks,0,d_streams[d]>>>(
				   d_boundingBoxes[d], objects.x, 
               d_primitives[d], objects.y,  
               d_lightInformation[d], objects.w, objects.z,
               d_materials[d], d_textures[d], 
				   d_randoms[d], origin, direction, angles, sceneInfo, 
				   postProcessingInfo, d_postProcessingBuffer[d], d_primitivesXYIds[d]);
			   break;
		   }
	   case vtFishEye:
		   {
			   k_fishEyeRenderer<<<grid,blocks,0,d_streams[d]>>>(
               d*size.y, d_nbGPUs,
				   d_boundingBoxes[d], objects.x, 
               d_primitives[d], objects.y,  
               d_lightInformation[d], objects.w, objects.z,
               d_materials[d], d_textures[d], 
				   d_randoms[d], origin, direction, angles, sceneInfo, 
				   postProcessingInfo, d_postProcessingBuffer[d], d_primitivesXYIds[d]);
			   break;
		   }
	   default:
		   {
			   k_standardRenderer<<<grid,blocks,0,d_streams[d]>>>(
               d*size.y, d_nbGPUs,
				   d_boundingBoxes[d], objects.x, 
               d_primitives[d], objects.y,  
               d_lightInformation[d], objects.w, objects.z,
               d_materials[d], d_textures[d], 
				   d_randoms[d], origin, direction, angles, sceneInfo, 
				   postProcessingInfo, d_postProcessingBuffer[d], d_primitivesXYIds[d]);
			   break;
		   }
	   }

	   cudaThreadSynchronize();
	   cudaError_t status = cudaGetLastError();
	   if(status != cudaSuccess) 
	   {
		   LOG_ERROR("ERROR: (" << status << ") " << cudaGetErrorString(status));
		   LOG_ERROR("INFO: Size(" << size.x << ", " << size.y << ")");
		   LOG_ERROR("INFO: Grid(" << grid.x << ", " << grid.y << ", " << grid.z <<")");
		   LOG_ERROR("nbActiveBoxes :" << objects.x);
		   LOG_ERROR("nbActivePrimitives :" << objects.y);
		   LOG_ERROR("nbActiveLamps :" << objects.z);
	   }

	   switch( postProcessingInfo.type.x )
	   {
	   case ppe_depthOfField:
		   k_depthOfField<<<grid,blocks,0,d_streams[d]>>>(
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[d],
			   d_randoms[d], 
			   d_bitmap[d] );
		   break;
	   case ppe_ambientOcclusion:
		   k_ambiantOcclusion<<<grid,blocks,0,d_streams[d]>>>(
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[d],
			   d_randoms[d], 
			   d_bitmap[d] );
		   break;
	   case ppe_cartoon:
		   k_cartoon<<<grid,blocks,0,d_streams[d]>>>(
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[d],
			   d_randoms[d], 
			   d_bitmap[d] );
		   break;
	   default:
		   k_default<<<grid,blocks,0,d_streams[d]>>>(
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[d],
            d_primitivesXYIds[d],
			   d_bitmap[d] );
		   break;
	   }
   }
}
