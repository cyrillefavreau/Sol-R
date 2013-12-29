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
#include <cuda_runtime.h>
//#include <helper_cuda.h>

// Project
#include "../Consts.h"
#include "../Logging.h"
#include "VectorUtils.cuh"
#include "GeometryIntersections.cuh"
#include "GeometryShaders.cuh"

// Device resources
#ifndef USE_MANAGED_MEMORY
BoundingBox*          d_boundingBoxes[MAX_GPU_COUNT]; 
Primitive*            d_primitives[MAX_GPU_COUNT];
#endif
Lamp*                 d_lamps[MAX_GPU_COUNT];
Material*             d_materials[MAX_GPU_COUNT];
BitmapBuffer*         d_textures[MAX_GPU_COUNT];
LightInformation*     d_lightInformation[MAX_GPU_COUNT];
RandomBuffer*         d_randoms[MAX_GPU_COUNT];
PostProcessingBuffer* d_postProcessingBuffer[MAX_GPU_COUNT];
BitmapBuffer*         d_bitmap[MAX_GPU_COUNT];
PrimitiveXYIdBuffer*  d_primitivesXYIds[MAX_GPU_COUNT];
cudaStream_t          d_streams[MAX_GPU_COUNT][MAX_STREAM_COUNT];

#define FREECUDARESOURCE(x) \
   if( x!=0 ) \
   { \
      checkCudaErrors(cudaFree(x)); \
      x=0; \
   }

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
intersection Vertex and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the intersection (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
__device__ inline float4 launchRay(
   const int& index,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	LightInformation* lightInformation, const int& lightInformationSize, const int& nbActiveLamps,
	Material*  materials, BitmapBuffer* textures,
	RandomBuffer*    randoms,
	const Ray&       ray, 
	const SceneInfo& sceneInfo,
	const PostProcessingInfo& postProcessingInfo,
	Vertex&          intersection,
	float&           depthOfField,
	PrimitiveXYIdBuffer& primitiveXYId)
{
	float4 intersectionColor   = {0.f,0.f,0.f,0.f};
	Vertex closestIntersection = {0.f,0.f,0.f};
   Vertex firstIntersection   = {0.f,0.f,0.f};
	Vertex normal              = {0.f,0.f,0.f};
	int    closestPrimitive  = -1;
	bool   carryon           = true;
	Ray    rayOrigin         = ray;
	float  initialRefraction = 1.f;
	int    iteration         = 0;
	primitiveXYId.x = -1;
   primitiveXYId.z = 0;
   primitiveXYId.w = 0;
   int currentMaterialId=-2;

   // TODO
   float  colorContributions[NB_MAX_ITERATIONS+1];
   float4 colors[NB_MAX_ITERATIONS+1];
   memset(&colorContributions[0],0,sizeof(float)*(NB_MAX_ITERATIONS+1));
   memset(&colors[0],0,sizeof(float4)*(NB_MAX_ITERATIONS+1));

	float4 recursiveBlinn = { 0.f, 0.f, 0.f, 0.f };

	// Variable declarations
	float  shadowIntensity = 0.f;
	float4 refractionFromColor;
	Vertex reflectedTarget;
   float4 closestColor = {0.f,0.f,0.f,0.f};
	float4 colorBox = {0.f,0.f,0.f,0.f};
	bool   back = false;

#ifdef PHOTON_ENERGY
   // Photon energy
   float photonDistance = sceneInfo.viewDistance.x;
   float previousTransparency = 1.f;
#endif // PHOTON_ENERGY

   // Reflected rays
   int reflectedRays=-1;
   Ray reflectedRay;
   float reflectedRatio;

   float4 rBlinn = {0.f,0.f,0.f,0.f};
   int currentMaxIteration = ( sceneInfo.graphicsLevel.x<3 ) ? 1 : sceneInfo.nbRayIterations.x+sceneInfo.pathTracingIteration.x;
   currentMaxIteration = (currentMaxIteration>NB_MAX_ITERATIONS) ? NB_MAX_ITERATIONS : currentMaxIteration;
#ifdef PHOTON_ENERGY
	while( iteration<currentMaxIteration && carryon && photonDistance>0.f ) 
#else
	while( iteration<currentMaxIteration && carryon ) 
#endif // PHOTON_ENERGY
	{
      Vertex areas = {0.f,0.f,0.f};
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
				normal, areas, closestColor, colorBox, back, currentMaterialId);
		}

		if( carryon ) 
		{
         currentMaterialId = primitives[closestPrimitive].materialId.x;

			if( iteration==0 )
			{
            colors[iteration].x = 0.f;
            colors[iteration].y = 0.f;
            colors[iteration].z = 0.f;
            colors[iteration].w = 0.f;
            colorContributions[iteration]=1.f;

				firstIntersection = closestIntersection;
            
            // Primitive ID for current pixel
            primitiveXYId.x = primitives[closestPrimitive].index.x;

			}

#ifdef PHOTON_ENERGY
         // Photon
         photonDistance -= length(closestIntersection-rayOrigin.origin) * (5.f-previousTransparency);
         previousTransparency = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].transparency.x;
#endif // PHOTON_ENERGY

			// Get object color
         colors[iteration] =
            primitiveShader( 
               index,
				   sceneInfo, postProcessingInfo,
				   boundingBoxes, nbActiveBoxes, 
			      primitives, nbActivePrimitives, 
               lightInformation, lightInformationSize, nbActiveLamps,
               materials, textures, 
               randoms, rayOrigin.origin, normal, 
               closestPrimitive, closestIntersection, areas, closestColor,
			      iteration, refractionFromColor, shadowIntensity, rBlinn );

         // Primitive illumination
         float colorLight=colors[iteration].x+colors[iteration].y+colors[iteration].z;
         primitiveXYId.z += 255*materials[currentMaterialId].innerIllumination.x;
         primitiveXYId.z += (colorLight>sceneInfo.transparentColor.x) ? 16 : 0;

         // ----------
			// Refraction
			// ----------
			if( materials[primitives[closestPrimitive].materialId.x].transparency.x != 0.f ) 
			{
				// Back of the object? If so, reset refraction to 1.f (air)
				float refraction = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].refraction.x;

				// Actual refraction
				Vertex O_E = normalize(rayOrigin.origin - closestIntersection);
				vectorRefraction( rayOrigin.direction, O_E, refraction, normal, initialRefraction );
				reflectedTarget = closestIntersection - rayOrigin.direction;

            colorContributions[iteration] = materials[primitives[closestPrimitive].materialId.x].transparency.x;
               
            // Prepare next ray
				initialRefraction = refraction;

				if( reflectedRays==-1 && materials[primitives[closestPrimitive].materialId.x].reflection.x != 0.f )
            {
					vectorReflection( reflectedRay.direction, O_E, normal );
					Vertex rt = closestIntersection - reflectedRay.direction;

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
					Vertex O_E = rayOrigin.origin - closestIntersection;
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
         rBlinn /= (iteration+1);
 			recursiveBlinn.x=(rBlinn.x>recursiveBlinn.x) ? rBlinn.x:recursiveBlinn.x;
 			recursiveBlinn.y=(rBlinn.y>recursiveBlinn.y) ? rBlinn.y:recursiveBlinn.y;
 			recursiveBlinn.z=(rBlinn.z>recursiveBlinn.z) ? rBlinn.z:recursiveBlinn.z;

         rayOrigin.origin    = closestIntersection + reflectedTarget*0.00001f; 
			rayOrigin.direction = reflectedTarget;

			// Noise management
			if( sceneInfo.pathTracingIteration.x != 0 && materials[primitives[closestPrimitive].materialId.x].color.w != 0.f)
			{
				// Randomize view
            float ratio = materials[primitives[closestPrimitive].materialId.x].color.w;
            ratio *= (materials[primitives[closestPrimitive].materialId.x].transparency.x==0.f) ? 1000.f : 1.f;
				int rindex = 3*sceneInfo.misc.y + sceneInfo.pathTracingIteration.x;
				rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x-3);
				rayOrigin.direction.x += randoms[rindex  ]*ratio;
				rayOrigin.direction.y += randoms[rindex+1]*ratio;
				rayOrigin.direction.z += randoms[rindex+2]*ratio;
			}
		}
		else
		{
#ifdef GRADIANT_BACKGROUND
         // Background
         Vertex normal = {0.f, 1.f, 0.f };
         Vertex dir = normalize(rayOrigin.direction-rayOrigin.origin);
         float angle = 0.5f*fabs(dot( normal, dir));
         angle = (angle>1.f) ? 1.f: angle;
			colors[iteration] = (1.f-angle)*sceneInfo.backgroundColor;
#else
			colors[iteration] = sceneInfo.backgroundColor;
#endif // GRADIANT_BACKGROUND
			colorContributions[iteration] = 1.f;
		}
		iteration++;
	}

   if( sceneInfo.graphicsLevel.x>=3 && reflectedRays != -1 ) // TODO: Draft mode should only test "sceneInfo.pathTracingIteration.x==iteration"
   {
      Vertex areas = {0.f,0.f,0.f};
      // TODO: Dodgy implementation		
      if( intersectionWithPrimitives(
			sceneInfo,
			boundingBoxes, nbActiveBoxes,
			primitives, nbActivePrimitives,
			materials, textures,
			reflectedRay,
			reflectedRays,  
			closestPrimitive, closestIntersection, 
         normal, areas, closestColor, colorBox, back, currentMaterialId) )
      {
         float4 color = primitiveShader( 
            index,
				sceneInfo, postProcessingInfo,
				boundingBoxes, nbActiveBoxes, 
			   primitives, nbActivePrimitives, 
            lightInformation, lightInformationSize, nbActiveLamps, 
            materials, textures, randoms, 
            reflectedRay.origin, normal, closestPrimitive, 
            closestIntersection, areas, closestColor, 
			   reflectedRays, 
            refractionFromColor, shadowIntensity, rBlinn );

         //colors[reflectedRays] = color*(1.f-reflectedRatio)+colors[reflectedRays]*reflectedRatio;
         colors[reflectedRays] += color*reflectedRatio;

         primitiveXYId.w = shadowIntensity*255;
      }
   }
   
   for( int i=iteration-2; i>=0; --i)
   {
      colors[i] = colors[i]*(1.f-colorContributions[i]) + colors[i+1]*colorContributions[i];
   }
   intersectionColor = colors[0];
	intersectionColor += recursiveBlinn;

	intersection = closestIntersection;

	float len = length(firstIntersection - ray.origin);
   if( closestPrimitive != -1 )
   {
      Primitive& primitive=primitives[closestPrimitive];
      if( materials[primitive.materialId.x].attributes.z==1 ) // Wireframe
      {
         len = sceneInfo.viewDistance.x;
      }
   }
#ifdef PHOTON_ENERGY
	// --------------------------------------------------
   // Photon energy
	// --------------------------------------------------
   intersectionColor *= ( photonDistance>0.f) ? (photonDistance/sceneInfo.viewDistance.x) : 0.f;
#endif // PHOTON_ENERGY

	// --------------------------------------------------
	// Fog
	// --------------------------------------------------
   //intersectionColor += randoms[((int)len + sceneInfo.misc.y)%100];

	// --------------------------------------------------
	// Background color
	// --------------------------------------------------
   float D1 = sceneInfo.viewDistance.x*0.95f;
   if( sceneInfo.misc.z==1 && len>D1)
   {
      float D2 = sceneInfo.viewDistance.x*0.05f;
      float a = len - D1;
      float b = 1.f-(a/D2);
      intersectionColor = intersectionColor*b + sceneInfo.backgroundColor*(1.f-b);
   }
   depthOfField = (len-depthOfField)/sceneInfo.viewDistance.x;

   // Primitive information
   primitiveXYId.y = iteration;

	// Depth of field
   intersectionColor -= colorBox;
	saturateVector( intersectionColor );
	return intersectionColor;
}

/*!
 * ------------------------------------------------------------------------------------------------------------------------
 * \brief      This kernel processes a "standard" image, meaning that the screen is a single image for 
 *             which every pixel is a ray of light entering the same camera.
 * ------------------------------------------------------------------------------------------------------------------------
 * \param[in]  occupancyParameters Contains the number of GPUs and streams involded in the GPU processing
 * \param[in]  device_split Y coordinate from where the current GPU should start working
 * \param[in]  stream_split Y coordinate from where the current stream should start working
 * \param[in]  BoundingBoxes Pointer to the array of bounding boxes
 * \param[in]  nbActiveBoxes Number of bounding boxes
 * \param[in]  primitives Pointer to the array of primitives
 * \param[in]  nbActivePrimitives Number of primitives
 * \param[in]  lightInformation Pointer to the array of light positions and intensities (Used for global illumination)
 * \param[in]  lightInformationSize Number of lights
 * \param[in]  nbActiveLamps Number of lamps
 * \param[in]  materials Pointer to the array of materials
 * \param[in]  textures Pointer to the array of textures
 * \param[in]  randoms Pointer to the array of random floats (GPUs are not good at generating numbers, done by the CPU)
 * \param[in]  origin Camera position
 * \param[in]  direction Camera LookAt
 * \param[in]  angles Angles applied to the camera. The rotation center is {0,0,0}
 * \param[in]  sceneInfo Information about the scene and environment
 * \param[in]  postProcessingInfo Information about PostProcessing effect
 * \param[out] postProcessingBuffer Pointer to the output array of color information
 * \param[out] primitiveXYIds Pointer to the array containing the Id of the primitivive for each pixel
 * ------------------------------------------------------------------------------------------------------------------------
*/
__global__ void k_standardRenderer(
   const int2   occupancyParameters,
   int          device_split,
   int          stream_split,
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
   Material*    materials,
	BitmapBuffer* textures,
	RandomBuffer* randoms,
	Vertex        origin,
	Vertex        direction,
	Vertex        angles,
	SceneInfo     sceneInfo,
	PostProcessingInfo postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	PrimitiveXYIdBuffer*  primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = (stream_split+y)*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   // And only process pixels that need extra rendering
   if(index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ||
     (sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y &&   // Still need to process iterations
      primitiveXYIds[index].w==0 &&                                 // Shadows? if so, compute soft shadows by randomizing light positions
      sceneInfo.pathTracingIteration.x>0 && 
      sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS)) return;

   Ray ray;
	ray.origin = origin;
	ray.direction = direction;

   Vertex rotationCenter = {0.f,0.f,0.f};
   if( sceneInfo.renderingType.x==vt3DVision)
   {
      rotationCenter = origin;
   }

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
		int rindex = 3*(index+sceneInfo.misc.y) + 5000;
		rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x-3);
		ray.origin.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.origin.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.origin.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);

      rindex = 3*(index+sceneInfo.misc.y);
		rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x-3);
		ray.direction.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.direction.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.direction.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
	}

	float dof = postProcessingInfo.param1.x;
	Vertex intersection;

   if( sceneInfo.misc.w == 1 ) // Isometric 3D
   {
      ray.direction.x = ray.origin.z*0.001f*(float)(x - (sceneInfo.width.x/2));
	   ray.direction.y = -ray.origin.z*0.001f*(float)(device_split+stream_split+y - (sceneInfo.height.x/2));
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
      ray.direction.y = ray.direction.y + step.y*(float)(device_split+stream_split+y - (sceneInfo.height.x/2));
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


   float4 color = {0.f,0.f,0.f,0.f};
   if( antialiasingActivated )
   {
      Ray r=ray;
	   for( int I=0; I<4; ++I )
	   {
         r.direction.x = ray.direction.x + AArotatedGrid[I].x;
         r.direction.y = ray.direction.y + AArotatedGrid[I].y;
	      float4 c = launchRay(
            index,
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
      index,
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

#ifdef ADVANCED_FEATURES
   // Randomize light intensity
	int rindex = index;
	rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
   color += sceneInfo.backgroundColor*randoms[rindex]*5.f;
#endif // ADVANCED_FEATURES
   
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

/*!
 * ------------------------------------------------------------------------------------------------------------------------
 * \brief      This kernel processes a fisheye image
 * ------------------------------------------------------------------------------------------------------------------------
 * \param[in]  occupancyParameters Contains the number of GPUs and streams involded in the GPU processing
 * \param[in]  device_split Y coordinate from where the current GPU should start working
 * \param[in]  stream_split Y coordinate from where the current stream should start working
 * \param[in]  BoundingBoxes Pointer to the array of bounding boxes
 * \param[in]  nbActiveBoxes Number of bounding boxes
 * \param[in]  primitives Pointer to the array of primitives
 * \param[in]  nbActivePrimitives Number of primitives
 * \param[in]  lightInformation Pointer to the array of light positions and intensities (Used for global illumination)
 * \param[in]  lightInformationSize Number of lights
 * \param[in]  nbActiveLamps Number of lamps
 * \param[in]  materials Pointer to the array of materials
 * \param[in]  textures Pointer to the array of textures
 * \param[in]  randoms Pointer to the array of random floats (GPUs are not good at generating numbers, done by the CPU)
 * \param[in]  origin Camera position
 * \param[in]  direction Camera LookAt
 * \param[in]  angles Angles applied to the camera. The rotation center is {0,0,0}
 * \param[in]  sceneInfo Information about the scene and environment
 * \param[in]  postProcessingInfo Information about PostProcessing effect
 * \param[out] postProcessingBuffer Pointer to the output array of color information
 * \param[out] primitiveXYIds Pointer to the array containing the Id of the primitivive for each pixel
 * ------------------------------------------------------------------------------------------------------------------------
*/
__global__ void k_fishEyeRenderer(
   const int2   occupancyParameters,
   int          split_y, 
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive*   primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*     materials,
	BitmapBuffer* textures,
	RandomBuffer* randoms,
	Vertex        origin,
	Vertex        direction,
	Vertex        angles,
	SceneInfo     sceneInfo,
	PostProcessingInfo    postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	PrimitiveXYIdBuffer*  primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   // And only process pixels that need extra rendering
   if(index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ||
     (sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y &&   // Still need to process iterations
      primitiveXYIds[index].w==0 &&                                 // Shadows? if so, compute soft shadows by randomizing light positions
      sceneInfo.pathTracingIteration.x>0 && 
      sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS)) return;

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
	Vertex intersection;

   // Normal Y axis
   float2 step;
   step.y=6400.f/(float)sceneInfo.height.x;
   ray.direction.y = ray.direction.y + step.y*(float)(split_y+y - (sceneInfo.height.x/2));

   // 360° X axis
   step.x = 2.f*PI/sceneInfo.width.x;
   step.y = 2.f*PI/sceneInfo.height.x;

   Vertex fishEyeAngles = {0.f,0.f,0.f};
   fishEyeAngles.y = angles.y + step.x*(float)x;
   //fishEyeAngles.x = angles.x + step.y*(float)y;

   vectorRotation( ray.direction, ray.origin, fishEyeAngles );

	//vectorRotation( ray.origin,    rotationCenter, angles );
	//vectorRotation( ray.direction, rotationCenter, angles );
	
   if( sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y && sceneInfo.pathTracingIteration.x>0 && sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS ) return;

   float4 color = {0.f,0.f,0.f,0.f};
   color += launchRay(
      index,
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

/*!
 * ------------------------------------------------------------------------------------------------------------------------
 * \brief      This kernel processes an anaglyph image. The sceneInfo.width3DVision parameter specifies the distance
 *             between both eyes.
 * ------------------------------------------------------------------------------------------------------------------------
 * \param[in]  occupancyParameters Contains the number of GPUs and streams involded in the GPU processing
 * \param[in]  BoundingBoxes Pointer to the array of bounding boxes
 * \param[in]  nbActiveBoxes Number of bounding boxes
 * \param[in]  primitives Pointer to the array of primitives
 * \param[in]  nbActivePrimitives Number of primitives
 * \param[in]  lightInformation Pointer to the array of light positions and intensities (Used for global illumination)
 * \param[in]  lightInformationSize Number of lights
 * \param[in]  nbActiveLamps Number of lamps
 * \param[in]  materials Pointer to the array of materials
 * \param[in]  textures Pointer to the array of textures
 * \param[in]  randoms Pointer to the array of random floats (GPUs are not good at generating numbers, done by the CPU)
 * \param[in]  origin Camera position
 * \param[in]  direction Camera LookAt
 * \param[in]  angles Angles applied to the camera. The rotation center is {0,0,0}
 * \param[in]  sceneInfo Information about the scene and environment
 * \param[in]  postProcessingInfo Information about PostProcessing effect
 * \param[out] postProcessingBuffer Pointer to the output array of color information
 * \param[out] primitiveXYIds Pointer to the array containing the Id of the primitivive for each pixel
 * ------------------------------------------------------------------------------------------------------------------------
*/
__global__ void k_anaglyphRenderer(
   const int2    occupancyParameters,
	BoundingBox*  boundingBoxes, int nbActiveBoxes,
	Primitive*    primitives, int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*     materials,
	BitmapBuffer* textures,
	RandomBuffer* randoms,
	Vertex        origin,
	Vertex        direction,
	Vertex        angles,
	SceneInfo     sceneInfo,
	PostProcessingInfo postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	PrimitiveXYIdBuffer*  primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   // And only process pixels that need extra rendering
   if(index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ||
     (sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y &&   // Still need to process iterations
      primitiveXYIds[index].w==0 &&                                 // Shadows? if so, compute soft shadows by randomizing light positions
      sceneInfo.pathTracingIteration.x>0 && 
      sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS)) return;

   Vertex rotationCenter = {0.f,0.f,0.f};
   if( sceneInfo.renderingType.x==vt3DVision)
   {
      rotationCenter = origin;
   }

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	Vertex intersection;
	Ray eyeRay;

   float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
   float2 step;
   step.x=ratio*6400.f/(float)sceneInfo.width.x;
   step.y=6400.f/(float)sceneInfo.height.x;

   // Left eye
	eyeRay.origin.x = origin.x - sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 colorLeft = launchRay(
      index,
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
	eyeRay.origin.x = origin.x + sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );
	
   float4 colorRight = launchRay(
      index,
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

/*!
 * ------------------------------------------------------------------------------------------------------------------------
 * \brief      This kernel processes two images in a side-by-side format. The sceneInfo.width3DVision parameter specifies
 *             the distance between both eyes.
 * ------------------------------------------------------------------------------------------------------------------------
 * \param[in]  occupancyParameters Contains the number of GPUs and streams involded in the GPU processing
 * \param[in]  BoundingBoxes Pointer to the array of bounding boxes
 * \param[in]  nbActiveBoxes Number of bounding boxes
 * \param[in]  primitives Pointer to the array of primitives
 * \param[in]  nbActivePrimitives Number of primitives
 * \param[in]  lightInformation Pointer to the array of light positions and intensities (Used for global illumination)
 * \param[in]  lightInformationSize Number of lights
 * \param[in]  nbActiveLamps Number of lamps
 * \param[in]  materials Pointer to the array of materials
 * \param[in]  textures Pointer to the array of textures
 * \param[in]  randoms Pointer to the array of random floats (GPUs are not good at generating numbers, done by the CPU)
 * \param[in]  origin Camera position
 * \param[in]  direction Camera LookAt
 * \param[in]  angles Angles applied to the camera. The rotation center is {0,0,0}
 * \param[in]  sceneInfo Information about the scene and environment
 * \param[in]  postProcessingInfo Information about PostProcessing effect
 * \param[out] postProcessingBuffer Pointer to the output array of color information
 * \param[out] primitiveXYIds Pointer to the array containing the Id of the primitivive for each pixel
 * ------------------------------------------------------------------------------------------------------------------------
*/
__global__ void k_3DVisionRenderer(
   const int2    occupancyParameters,
	BoundingBox*  boundingBoxes, int nbActiveBoxes,
	Primitive*    primitives,    int nbActivePrimitives,
   LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
	Material*     materials,
	BitmapBuffer* textures,
	RandomBuffer* randoms,
	Vertex        origin,
	Vertex        direction,
	Vertex        angles,
	SceneInfo     sceneInfo,
	PostProcessingInfo    postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	PrimitiveXYIdBuffer*  primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   // And only process pixels that need extra rendering
   if(index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ||
     (sceneInfo.pathTracingIteration.x>primitiveXYIds[index].y &&   // Still need to process iterations
      primitiveXYIds[index].w==0 &&                                 // Shadows? if so, compute soft shadows by randomizing light positions
      sceneInfo.pathTracingIteration.x>0 && 
      sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS)) return;

   float focus = fabs(postProcessingBuffer[sceneInfo.width.x/2*sceneInfo.height.x/2].w - origin.z);
   float eyeSeparation = sceneInfo.width3DVision.x*(direction.z/focus);

   Vertex rotationCenter = {0.f,0.f,0.f};
   if( sceneInfo.renderingType.x==vt3DVision)
   {
      rotationCenter = origin;
   }

   if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	Vertex intersection;
	int halfWidth  = sceneInfo.width.x/2;

   float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
   float2 step;
   step.x=ratio*6400.f/(float)sceneInfo.width.x;
   step.y=6400.f/(float)sceneInfo.height.x;

   Ray eyeRay;
	if( x<halfWidth ) 
	{
		// Left eye
		eyeRay.origin.x = origin.x + eyeSeparation;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

      eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2) + halfWidth/2 ) + sceneInfo.width3DVision.x;
		eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}
	else
	{
		// Right eye
		eyeRay.origin.x = origin.x - eyeSeparation;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

		eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.width.x/2) - halfWidth/2) - sceneInfo.width3DVision.x;
		eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}

   //if(sqrt(eyeRay.direction.x*eyeRay.direction.x+eyeRay.direction.y*eyeRay.direction.y)>(halfWidth*6)) return;

	vectorRotation( eyeRay.origin,    rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 color = launchRay(
      index,
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

#ifdef ADVANCED_FEATURES
   // Randomize light intensity
	int rindex = index;
	rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
   color += sceneInfo.backgroundColor*randoms[rindex]*5.f;
#endif // ADVANCED_FEATURES

   // Contribute to final image
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

/*!
 * ------------------------------------------------------------------------------------------------------------------------
 * \brief      This post-processing kernel simply converts the contents of the postProcessingBuffer into a bitmap
 * ------------------------------------------------------------------------------------------------------------------------
 * \param[in]  occupancyParameters Contains the number of GPUs and streams involded in the GPU processing
 * \param[in]  sceneInfo Information about the scene and environment
 * \param[in]  postProcessingInfo Information about PostProcessing effect
 * \param[in]  postProcessingBuffer Pointer to the output array of color information
 * \param[out] Bitmap Pointer to a bitmap. The bitmap is encoded according to the value of the sceneInfo.misc.x parameter
 * ------------------------------------------------------------------------------------------------------------------------
*/
__global__ void k_default(
   const int2            occupancyParameters,
	SceneInfo             sceneInfo,
	PostProcessingInfo    PostProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	BitmapBuffer*         bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   if( index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ) return;

#if 1
   float4 localColor = postProcessingBuffer[index];
   if(sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
   {
      localColor /= (float)(sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
   }
#else
   float4 localColor;
   localColor.x = float(index)/float(sceneInfo.width.x*sceneInfo.height.x);
   localColor.y = 0.f;
   localColor.z = 0.f;
   localColor.w = 0.f;
#endif // 0

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Depth of field
________________________________________________________________________________
*/
__global__ void k_depthOfField(
   const int2            occupancyParameters,
	SceneInfo             sceneInfo,
	PostProcessingInfo    postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	RandomBuffer*         randoms,
	BitmapBuffer*         bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   if( index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ) return;
   
   float  depth = postProcessingInfo.param2.x*postProcessingBuffer[index].w;
	int    wh = sceneInfo.width.x*sceneInfo.height.x;

   float4 localColor = {0.f,0.f,0.f};
	for( int i=0; i<postProcessingInfo.param3.x; ++i )
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
	localColor /= postProcessingInfo.param3.x;

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
   const int2            occupancyParameters,
	SceneInfo             sceneInfo,
	PostProcessingInfo    postProcessingInfo,
	PostProcessingBuffer* postProcessingBuffer,
	RandomBuffer*         randoms,
	BitmapBuffer*         bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   if( index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ) return;

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

Post Processing Effect: Radiosity
________________________________________________________________________________
*/
__global__ void k_radiosity(
   const int2            occupancyParameters,
	SceneInfo             sceneInfo,
	PostProcessingInfo    postProcessingInfo,
   PrimitiveXYIdBuffer*  primitiveXYIds,
	PostProcessingBuffer* postProcessingBuffer,
	RandomBuffer*         randoms,
	BitmapBuffer*         bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   if( index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ) return;
   
   int    wh = sceneInfo.width.x*sceneInfo.height.x;

   int div = (sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS) ? (sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1) : 1;

   float4 localColor = {0.f,0.f,0.f};
	for( int i=0; i<postProcessingInfo.param3.x; ++i )
	{
		int ix = (i+sceneInfo.misc.y+sceneInfo.pathTracingIteration.x)%wh;
		int iy = (i+sceneInfo.misc.y+sceneInfo.width.x)%wh;
		int xx = x+randoms[ix]*postProcessingInfo.param2.x;
		int yy = y+randoms[iy]*postProcessingInfo.param2.x;
		localColor += postProcessingBuffer[index];
		if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
		{
			int localIndex = yy*sceneInfo.width.x+xx;
			localColor += ( localIndex>=0 && localIndex<wh ) ? div*primitiveXYIds[localIndex].z/255 : 0.f;
      }
	}
	localColor /= postProcessingInfo.param3.x;
   localColor /= div;

	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Radiosity
________________________________________________________________________________
*/
__global__ void k_oneColor(
   const int2            occupancyParameters,
	SceneInfo             sceneInfo,
	PostProcessingBuffer* postProcessingBuffer,
	BitmapBuffer*         bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   // Beware out of bounds error! \[^_^]/
   if( index>=sceneInfo.width.x*sceneInfo.height.x/occupancyParameters.x ) return;
   int div = (sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS) ? (sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1) : 1;

   float4 localColor = postProcessingBuffer[index]/div;
   if(fabs(localColor.x-sceneInfo.backgroundColor.x)>sceneInfo.transparentColor.x || 
      fabs(localColor.y-sceneInfo.backgroundColor.y)>sceneInfo.transparentColor.x ||
      fabs(localColor.z-sceneInfo.backgroundColor.z)>sceneInfo.transparentColor.x )
   {
      float totalColor = (localColor.x+localColor.y+localColor.z)/3.f;
      localColor.x = totalColor;
      localColor.y = totalColor;
      localColor.z = totalColor;
   }
   //localColor /= div;
	localColor.w = 1.f;
	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

GPU initialization
________________________________________________________________________________
*/
extern "C" void initialize_scene( 
   int2             occupancyParameters,
   SceneInfo sceneInfo,
   int        nbPrimitives,
   int        nbLamps,
   int        nbMaterials
#ifdef USE_MANAGED_MEMORY
   ,BoundingBox*&    boundingBoxes
   ,Primitive*&      primitives
#endif
   )
{
   // Multi GPU initialization
   int nbGPUs;
   checkCudaErrors(cudaGetDeviceCount(&nbGPUs));
   if(nbGPUs > MAX_GPU_COUNT)
   {
	   nbGPUs = MAX_GPU_COUNT;
   }
   if( occupancyParameters.x > nbGPUs )
   {
      LOG_INFO(1 ,"You asked for " << occupancyParameters.x << " CUDA-capable devices, but only " << nbGPUs << " are available");
      occupancyParameters.x = nbGPUs;
   }
   else
   {
      LOG_INFO(1 ,"CUDA-capable device count: " << occupancyParameters.x);
   }

   for( int device(0); device<occupancyParameters.x; ++device )
   {
      int totalMemoryAllocation(0);
      checkCudaErrors(cudaSetDevice(device));
      for( int stream(0); stream<occupancyParameters.y; ++stream )
      {
         checkCudaErrors(cudaStreamCreate(&d_streams[device][stream]));
      }
      LOG_INFO(1, "Created " << occupancyParameters.y << " streams on device " << device );

      // Scene resources
      // Scene resources
      int size(NB_MAX_BOXES*sizeof(BoundingBox));

#ifdef USE_MANAGED_MEMORY
	   checkCudaErrors(cudaMallocManaged( &boundingBoxes, size, cudaMemAttachHost));
#else
      checkCudaErrors(cudaMalloc( (void**)&d_boundingBoxes[device], size));
#endif 
      LOG_INFO( 1, "d_boundingBoxes: " << size << " bytes" );
      totalMemoryAllocation += size;

      size=NB_MAX_PRIMITIVES*sizeof(Primitive);
#ifdef USE_MANAGED_MEMORY
	   checkCudaErrors(cudaMallocManaged( &primitives, size, cudaMemAttachHost));
#else
      checkCudaErrors(cudaMalloc( (void**)&d_primitives[device], size));
#endif 
      LOG_INFO( 1, "d_primitives: " << size << " bytes" );
      totalMemoryAllocation += size;
	   
      size=NB_MAX_LAMPS*sizeof(Lamp);
      checkCudaErrors(cudaMalloc( (void**)&d_lamps[device], size));
      LOG_INFO(3, "d_lamps: " << size << " bytes" );
      totalMemoryAllocation += size;
	   
      size=NB_MAX_MATERIALS*sizeof(Material);
      checkCudaErrors(cudaMalloc( (void**)&d_materials[device], size));
      LOG_INFO(3, "d_materials: " << size << " bytes" );
      totalMemoryAllocation += size;
	   
      size=NB_MAX_LIGHTINFORMATIONS*sizeof(LightInformation);
      checkCudaErrors(cudaMalloc( (void**)&d_lightInformation[device], size));
      LOG_INFO(3, "d_lightInformation: " << size << " bytes" );
      totalMemoryAllocation += size;
	   
      size=sceneInfo.width.x*sceneInfo.height.x*sizeof(RandomBuffer);
      checkCudaErrors(cudaMalloc( (void**)&d_randoms[device], size));
      LOG_INFO(3, "d_randoms: " << size << " bytes" );
      totalMemoryAllocation += size;

	   // Rendering canvas
      size = sceneInfo.width.x*sceneInfo.height.x*sizeof(PostProcessingBuffer)/occupancyParameters.x;
	   checkCudaErrors(cudaMalloc( (void**)&d_postProcessingBuffer[device], size));
      LOG_INFO(3, "d_postProcessingBuffer: " << size << " bytes" );
      totalMemoryAllocation += size;

      size = sceneInfo.width.x*sceneInfo.height.x*gColorDepth*sizeof(BitmapBuffer)/occupancyParameters.x;
	   checkCudaErrors(cudaMalloc( (void**)&d_bitmap[device], size));
      LOG_INFO(3, "d_bitmap: " << size << " bytes" );
      totalMemoryAllocation += size;

      size = sceneInfo.width.x*sceneInfo.height.x*sizeof(PrimitiveXYIdBuffer)/occupancyParameters.x;
      checkCudaErrors(cudaMalloc( (void**)&d_primitivesXYIds[device], size));
      LOG_INFO(3, "d_primitivesXYIds: " << size << " bytes" );
      totalMemoryAllocation += size;

      d_textures[device] = nullptr;
      LOG_INFO(3, "Total GPU memory allocated on device " << device << ": " << totalMemoryAllocation << " bytes" );
   }
	LOG_INFO( 3, "GPU: SceneInfo         : " << sizeof(SceneInfo) );
	LOG_INFO( 3, "GPU: Ray               : " << sizeof(Ray) );
	LOG_INFO( 3, "GPU: PrimitiveType     : " << sizeof(PrimitiveType) );
	LOG_INFO( 3, "GPU: Material          : " << sizeof(Material) );
	LOG_INFO( 3, "GPU: BoundingBox       : " << sizeof(BoundingBox) );
	LOG_INFO( 3, "GPU: Primitive         : " << sizeof(Primitive) );
	LOG_INFO( 3, "GPU: PostProcessingType: " << sizeof(PostProcessingType) );
	LOG_INFO( 3, "GPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) );
	LOG_INFO( 3, "Textures " << NB_MAX_TEXTURES );
}

/*
________________________________________________________________________________

GPU finalization
________________________________________________________________________________
*/
extern "C" void finalize_scene( 
   int2 occupancyParameters
#ifdef USE_MANAGED_MEMORY
	,BoundingBox* boundingBoxes
   ,Primitive*   primitives
#endif
   )
{
   LOG_INFO(1 ,"Releasing device resources");
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
#ifdef USE_MANAGED_MEMORY
	   FREECUDARESOURCE(boundingBoxes);
	   FREECUDARESOURCE(primitives);
#else
	   FREECUDARESOURCE(d_boundingBoxes[device]);
	   FREECUDARESOURCE(d_primitives[device]);
#endif
      FREECUDARESOURCE(d_lamps[device]);
	   FREECUDARESOURCE(d_materials[device]);
      FREECUDARESOURCE(d_textures[device]);
      FREECUDARESOURCE(d_lightInformation[device]);
	   FREECUDARESOURCE(d_randoms[device]);
	   FREECUDARESOURCE(d_postProcessingBuffer[device]);
      FREECUDARESOURCE(d_bitmap[device]);
	   FREECUDARESOURCE(d_primitivesXYIds[device]);
      for( int stream(0); stream<occupancyParameters.y; ++stream )
      {
         checkCudaErrors(cudaStreamDestroy(d_streams[device][stream]));
         d_streams[device][stream] = 0;
      }
      cudaDeviceReset();
   }
}

/*
________________________________________________________________________________

CPU -> GPU data transfers
________________________________________________________________________________
*/
extern "C" void h2d_scene( 
   int2   occupancyParameters,
   BoundingBox* boundingBoxes, 
   int    nbActiveBoxes,
	Primitive*   primitives, 
   int    nbPrimitives,
	Lamp*        lamps, 
   int    nbLamps )
{
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
#ifndef USE_MANAGED_MEMORY
	   checkCudaErrors(cudaMemcpyAsync( d_boundingBoxes[device],      boundingBoxes,      nbActiveBoxes*sizeof(BoundingBox), cudaMemcpyHostToDevice, d_streams[device][0] ));
	   checkCudaErrors(cudaMemcpyAsync( d_primitives[device],         primitives,         nbPrimitives*sizeof(Primitive),    cudaMemcpyHostToDevice, d_streams[device][0] ));
#endif
	   checkCudaErrors(cudaMemcpyAsync( d_lamps[device],              lamps,              nbLamps*sizeof(Lamp),              cudaMemcpyHostToDevice, d_streams[device][0] ));
   }
}

extern "C" void h2d_materials( 
   int2 occupancyParameters,
	Material*  materials, 
   int  nbActiveMaterials,
	float*     randoms,   
   int  nbRandoms)
{
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
	   checkCudaErrors(cudaMemcpyAsync( d_materials[device], materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice, d_streams[device][0] ));
	   checkCudaErrors(cudaMemcpyAsync( d_randoms[device],   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice, d_streams[device][0] ));
   }
}

extern "C" void h2d_textures( 
   int2          occupancyParameters,
	int           activeTextures,
   TextureInformation* textureInfos )
{
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
      int totalSize(0);
      for( int i(0); i<activeTextures; ++i )
      {
         if( textureInfos[i].buffer )
         {
            //LOG_INFO(3, "Texture [" << i << "] memory allocated=" << textureInfos[i].size.x*textureInfos[i].size.y*textureInfos[i].size.z << " bytes" );
            totalSize += textureInfos[i].size.x*textureInfos[i].size.y*textureInfos[i].size.z;
         }
      }
      FREECUDARESOURCE(d_textures[device]);
      if( totalSize>0 )
      {
         totalSize *= sizeof(BitmapBuffer);
         LOG_INFO(3, "Total GPU texture memory allocated: " << totalSize << " bytes" );
	      checkCudaErrors(cudaMalloc( (void**)&d_textures[device], totalSize));

         for( int i(0); i<activeTextures; ++i )
         {
            if( textureInfos[i].buffer != nullptr )
            {
               LOG_INFO( 3, "Texture [" << i << "] transfered=" << textureInfos[i].size.x << "," << textureInfos[i].size.y << "," << textureInfos[i].size.z << ", offset=" << textureInfos[i].offset );
               int textureSize = textureInfos[i].size.x*textureInfos[i].size.y*textureInfos[i].size.z;
               checkCudaErrors(cudaMemcpyAsync( d_textures[device]+textureInfos[i].offset, textureInfos[i].buffer, textureSize*sizeof(BitmapBuffer), cudaMemcpyHostToDevice, d_streams[device][0] ));
            }
         }
      }
   }
}

extern "C" void h2d_lightInformation( 
   int2        occupancyParameters,
   LightInformation* lightInformation, 
   int         lightInformationSize)
{
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
	   checkCudaErrors(cudaMemcpyAsync( d_lightInformation[device],  lightInformation,  lightInformationSize*sizeof(LightInformation), cudaMemcpyHostToDevice, d_streams[device][0] ));
   }
}

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   int2    occupancyParameters,
	BitmapBuffer* kinectVideo, 
   BitmapBuffer* kinectDepth )
{
   for( int device(0); device<occupancyParameters.x; ++device )
   {
	   checkCudaErrors(cudaMemcpyAsync( d_textures[device],                  kinectVideo, gKinectVideoSize*sizeof(BitmapBuffer), cudaMemcpyHostToDevice, d_streams[device][0] ));
	   checkCudaErrors(cudaMemcpyAsync( d_textures[device]+gKinectVideoSize, kinectDepth, gKinectDepthSize*sizeof(BitmapBuffer), cudaMemcpyHostToDevice, d_streams[device][0] ));
   }
}
#endif // USE_KINECT

/*
________________________________________________________________________________

GPU -> CPU data transfers
________________________________________________________________________________
*/
extern "C" void d2h_bitmap( 
   int2           occupancyParameters,
   SceneInfo      sceneInfo,
   BitmapBuffer*        bitmap, 
   PrimitiveXYIdBuffer* primitivesXYIds )
{
   int offsetBitmap = sceneInfo.width.x*sceneInfo.height.x*gColorDepth*sizeof(BitmapBuffer)/occupancyParameters.x;
   int offsetXYIds  = sceneInfo.width.x*sceneInfo.height.x*sizeof(PrimitiveXYIdBuffer)/occupancyParameters.x;
   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));
      
      // Synchronize stream
      for( int stream(0); stream<occupancyParameters.y; ++stream )
      {
         LOG_INFO(3, "Synchronizing stream " << stream << "/" << occupancyParameters.y << " on device " << device << "/" << occupancyParameters.x );
         checkCudaErrors(cudaStreamSynchronize(d_streams[device][stream]));
      }

      // Copy results back to host
      LOG_INFO(3, "Copy results back to host: " << device*offsetBitmap << "/" << offsetBitmap << ", " << device*offsetXYIds << "/" << offsetXYIds );
      checkCudaErrors(cudaMemcpyAsync( bitmap+device*offsetBitmap,         d_bitmap[device],          offsetBitmap, cudaMemcpyDeviceToHost));
	   checkCudaErrors(cudaMemcpyAsync( primitivesXYIds+device*offsetXYIds, d_primitivesXYIds[device], offsetXYIds,  cudaMemcpyDeviceToHost));
   }
}

/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
extern "C" void cudaRender(
   int2               occupancyParameters,
	int4               blockSize,
	SceneInfo          sceneInfo,
	int4               objects,
	PostProcessingInfo postProcessingInfo,
	Vertex             origin,
	Vertex             direction,
	Vertex             angles
#ifdef USE_MANAGED_MEMORY
   ,BoundingBox*            boundingBoxes
	,Primitive*              primitives
#endif
   )
{
   LOG_INFO(3, "CPU PostProcessingBuffer: " << sizeof(PostProcessingBuffer));
   LOG_INFO(3, "CPU PrimitiveXYIdBuffer : " << sizeof(PrimitiveXYIdBuffer));
   LOG_INFO(3, "CPU BoundingBox         : " << sizeof(BoundingBox));
   LOG_INFO(3, "CPU Primitive           : " << sizeof(Primitive));
   LOG_INFO(3, "CPU Material            : " << sizeof(Material));

	int2 size;
	size.x = static_cast<int>(sceneInfo.width.x);
	size.y = static_cast<int>(sceneInfo.height.x) / (occupancyParameters.x*occupancyParameters.y);

   dim3 grid;
   grid.x = (size.x+blockSize.x-1)/blockSize.x;
   grid.y = (size.y+blockSize.y-1)/blockSize.y;
   grid.z = 1;
	
   dim3 blocks;
   blocks.x = blockSize.x;
   blocks.y = blockSize.y;
   blocks.z = blockSize.z;

   std::string step("Initialization");

   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));

      for( int stream(0); stream<occupancyParameters.y; ++stream )
      {
	      switch( sceneInfo.renderingType.x ) 
	      {
	      case vtAnaglyph:
		      {
               step = "vtAnaglyph";
			      k_anaglyphRenderer<<<grid,blocks,0,d_streams[device][stream]>>>(
                  occupancyParameters,
#ifndef USE_MANAGED_MEMORY
				      d_boundingBoxes[device],
#else
				      boundingBoxes, 
#endif
                  objects.x,
#ifndef USE_MANAGED_MEMORY
                  d_primitives[device], 
#else
				      primitives,
#endif
                  objects.y,  
                  d_lightInformation[device], objects.w, objects.z,
                  d_materials[device], d_textures[device], 
				      d_randoms[device], origin, direction, angles, sceneInfo, 
				      postProcessingInfo, d_postProcessingBuffer[device], d_primitivesXYIds[device]);
			      break;
		      }
         case vt3DVision:
		      {
               step = "vt3DVision";
			      k_3DVisionRenderer<<<grid,blocks,0,d_streams[device][stream]>>>(
                  occupancyParameters,
#ifndef USE_MANAGED_MEMORY
				      d_boundingBoxes[device],
#else
				      boundingBoxes, 
#endif
                  objects.x, 
#ifndef USE_MANAGED_MEMORY
                  d_primitives[device], 
#else
				      primitives,
#endif
                  objects.y,  
                  d_lightInformation[device], objects.w, objects.z,
                  d_materials[device], d_textures[device], 
				      d_randoms[device], origin, direction, angles, sceneInfo, 
				      postProcessingInfo, d_postProcessingBuffer[device], d_primitivesXYIds[device]);
			      break;
		      }
	      case vtFishEye:
		      {
               step = "vtFishEye";
			      k_fishEyeRenderer<<<grid,blocks,0,d_streams[device][stream]>>>(
                  occupancyParameters,
                  device*stream*size.y,
#ifndef USE_MANAGED_MEMORY
				      d_boundingBoxes[device],
#else
				      boundingBoxes, 
#endif
                  objects.x, 
#ifndef USE_MANAGED_MEMORY
                  d_primitives[device], 
#else
				      primitives,
#endif
                  objects.y,  
                  d_lightInformation[device], objects.w, objects.z,
                  d_materials[device], d_textures[device], 
				      d_randoms[device], origin, direction, angles, sceneInfo, 
				      postProcessingInfo, d_postProcessingBuffer[device], d_primitivesXYIds[device]);
			      break;
		      }
	      default:
		      {
               step = "k_standardRenderer";
			      k_standardRenderer<<<grid,blocks,0,d_streams[device][stream]>>>(
                  occupancyParameters,
                  device*(sceneInfo.height.x/occupancyParameters.x),
                  stream*size.y,
#ifndef USE_MANAGED_MEMORY
				      d_boundingBoxes[device],
#else
				      boundingBoxes, 
#endif
				      objects.x, 
#ifndef USE_MANAGED_MEMORY
                  d_primitives[device], 
#else
				      primitives,
#endif
                  objects.y,  
                  d_lightInformation[device], objects.w, objects.z,
                  d_materials[device], d_textures[device], 
				      d_randoms[device], origin, direction, angles, sceneInfo, 
				      postProcessingInfo, d_postProcessingBuffer[device], d_primitivesXYIds[device]);
			      break;
		      }
         }
	      cudaError_t status = cudaGetLastError();
	      if(status != cudaSuccess) 
	      {
		      LOG_ERROR("********************************************************************************");
            LOG_ERROR("Error code : [" << status << "] " << cudaGetErrorString(status));
		      LOG_ERROR("Step       : " << step);
		      LOG_ERROR("Device     : " << device);
		      LOG_ERROR("Stream     : " << stream);
		      LOG_ERROR("Image size : " << size.x << ", " << size.y);
		      LOG_ERROR("Grid size  : " << grid.x << ", " << grid.y << ", " << grid.z);
		      LOG_ERROR("Block size : " << blocks.x << ", " << blocks.y << ", " << blocks.z);
		      LOG_ERROR("Boxes      : " << objects.x);
		      LOG_ERROR("Primitives : " << objects.y);
		      LOG_ERROR("Lamps      : " << objects.z);
		      LOG_ERROR("********************************************************************************");
	      }
	   }
      //step = "cudaThreadSynchronize";
	   //checkCudaErrors(cudaThreadSynchronize());
   }

   // --------------------------------------------------------------------------------
   // Post processing on device 0, stream 0
   // --------------------------------------------------------------------------------
	size.x = static_cast<int>(sceneInfo.width.x);
	size.y = static_cast<int>(sceneInfo.height.x) / occupancyParameters.x;

   grid.x = (size.x+blockSize.x-1)/blockSize.x;
   grid.y = (size.y+blockSize.y-1)/blockSize.y;
   grid.z = 1;
	
   blocks.x = blockSize.x;
   blocks.y = blockSize.y;
   blocks.z = blockSize.z;

   for( int device(0); device<occupancyParameters.x; ++device )
   {
      checkCudaErrors(cudaSetDevice(device));

	   switch( postProcessingInfo.type.x )
	   {
	   case ppe_depthOfField:
         step = "ppe_depthOfField";
		   k_depthOfField<<<grid,blocks,0,d_streams[device][0]>>>(
            occupancyParameters,
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[device],
			   d_randoms[device], 
			   d_bitmap[device] );
		   break;
	   case ppe_ambientOcclusion:
         step = "ppe_ambientOcclusion";
		   k_ambiantOcclusion<<<grid,blocks,0,d_streams[device][0]>>>(
            occupancyParameters,
			   sceneInfo, 
			   postProcessingInfo, 
			   d_postProcessingBuffer[device],
			   d_randoms[device], 
			   d_bitmap[device] );
		   break;
	   case ppe_radiosity:
         step = "ppe_enlightment";
		   k_radiosity<<<grid,blocks,0,d_streams[device][0]>>>(
            occupancyParameters,
			   sceneInfo, 
			   postProcessingInfo, 
            d_primitivesXYIds[device],
			   d_postProcessingBuffer[device],
			   d_randoms[device], 
			   d_bitmap[device] );
		   break;
	   case ppe_oneColor:
		   k_oneColor<<<grid,blocks,0,d_streams[device][0]>>>(
            occupancyParameters,
			   sceneInfo, 
			   d_postProcessingBuffer[device],
			   d_bitmap[device] );
		   break;
	   default:
         step = "k_default";
         k_default<<<grid,blocks,0,d_streams[device][0]>>>(
            occupancyParameters,
            sceneInfo,
			   postProcessingInfo, 
			   d_postProcessingBuffer[device],
			   d_bitmap[device] );
		   break;
	   }

      cudaError_t status = cudaGetLastError();
	   if(status != cudaSuccess) 
	   {
		   LOG_ERROR("********************************************************************************");
         LOG_ERROR("Error code : [" << status << "] " << cudaGetErrorString(status));
		   LOG_ERROR("Step       : " << step);
		   LOG_ERROR("Device     : " << device);
		   LOG_ERROR("Stream     : " << 0);
		   LOG_ERROR("Image size : " << size.x << ", " << size.y);
		   LOG_ERROR("Grid size  : " << grid.x << ", " << grid.y << ", " << grid.z);
		   LOG_ERROR("Block size : " << blocks.x << ", " << blocks.y << ", " << blocks.z);
		   LOG_ERROR("Boxes      : " << objects.x);
		   LOG_ERROR("Primitives : " << objects.y);
		   LOG_ERROR("Lamps      : " << objects.z);
		   LOG_ERROR("********************************************************************************");
	   }
   }
}
