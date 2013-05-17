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

#pragma once

// Project
#include "../Consts.h"
#include "CudaDataTypes.h"
#include "TextureMapping.cuh"
#include "GeometryIntersections.cuh"

// Cuda
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

/*
________________________________________________________________________________

Convert float3 into OpenGL RGB color
________________________________________________________________________________
*/
__device__ void makeColor(
	const SceneInfo& sceneInfo,
	float4&   color,
	char*     bitmap,
	int       index)
{
   int mdc_index = index*gColorDepth; 
	color.x = (color.x>1.f) ? 1.f : color.x;
	color.y = (color.y>1.f) ? 1.f : color.y; 
	color.z = (color.z>1.f) ? 1.f : color.z;

   switch( sceneInfo.misc.x )
   {
      case otOpenGL: 
	   {
		   // OpenGL
		   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
         break;
	   }
      case otDelphi: 
	   {
		   // Delphi
		   bitmap[mdc_index  ] = (char)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (char)(color.x*255.f); // Red
         break;
	   }
      case otJPEG: 
	   {
         mdc_index = (sceneInfo.width.x*sceneInfo.height.x-index)*gColorDepth; 
		   // JPEG
		   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
         break;
	   }
   }
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray 
which origin is the considered 3D float3 and which direction is defined by the 
light source center.
.
. * Lamp                     Ray = Origin -> Light Source Center
.  \
.   \##
.   #### object
.    ##
.      \
.       \  Origin
.--------O-------
.
@return 1.f when pixel is in the shades

________________________________________________________________________________
*/
__device__ float processShadows(
	const SceneInfo& sceneInfo,
	BoundingBox*  boudingBoxes, const int& nbActiveBoxes,
	Primitive*    primitives,
	Material*     materials,
	char*         textures,
	const int&    nbPrimitives, 
	const float3& lampCenter, 
	const float3& origin, 
	const int&    objectId,
	const int&    iteration,
   float4&       color)
{
	float result = 0.f;
	int cptBoxes = 0;
   color.x = 0.f;
   color.y = 0.f;
   color.z = 0.f;
   int it=-1;
	Ray r;
	r.origin    = origin;
	r.direction = lampCenter-origin;
	computeRayAttributes( r );

   while( result<sceneInfo.shadowIntensity.x && cptBoxes<nbActiveBoxes )
	{

		BoundingBox& box = boudingBoxes[cptBoxes];
		if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x))
		{
			int cptPrimitives = 0;
			while( result<sceneInfo.shadowIntensity.x && cptPrimitives<box.nbPrimitives.x)
			{
				float3 intersection = {0.f,0.f,0.f};
				float3 normal       = {0.f,0.f,0.f};
				float  shadowIntensity = 0.f;

				if( (box.startIndex.x+cptPrimitives) != objectId )
				{
					Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];

					bool hit = false;
					bool back;
					switch(primitive.type.x)
					{
					case ptSphere   : hit=sphereIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); break;
               case ptEllipsoid: hit=ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); break;
					case ptCylinder :	hit=cylinderIntersection ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); break;
					case ptTriangle :	hit=triangleIntersection ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); break;
					case ptCamera   : hit=false; break;
					default         : hit=planeIntersection    ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false ); break;
					}

					if( hit )
					{
						float3 O_I = intersection-r.origin;
						float3 O_L = r.direction;
						float l = length(O_I);
						if( l>EPSILON && l<length(O_L) )
						{
                     float ratio = 0.f;
                     if( materials[primitive.materialId.x].transparency.x != 0.f )
                     {
                        O_L=normalize(O_L);
                        float a=fabs(dot(O_L,normal));
                        float r = (materials[primitive.materialId.x].transparency.x == 0.f ) ? 1.f : (1.f-0.8f*materials[primitive.materialId.x].transparency.x);
                        ratio = r*a*shadowIntensity*sceneInfo.shadowIntensity.x;
                        // Shadow color
                        color.x  += ratio*(0.3f-0.3f*materials[primitive.materialId.x].color.x);
                        color.y  += ratio*(0.3f-0.3f*materials[primitive.materialId.x].color.y);
                        color.z  += ratio*(0.3f-0.3f*materials[primitive.materialId.x].color.z);
                     }
                     else
                     {
                        float r = (materials[primitive.materialId.x].transparency.x == 0.f ) ? 1.f : (1.f-materials[primitive.materialId.x].transparency.x);
                        ratio = r*shadowIntensity*sceneInfo.shadowIntensity.x;
                        // Shadow color
                        color.x  += ratio;
                        color.y  += ratio;
                        color.z  += ratio;
                     }
                     result += ratio;
                  }
                  it++;
					}
				}
				cptPrimitives++;
			}
		}
		cptBoxes++;
	}
	result = (result>sceneInfo.shadowIntensity.x) ? sceneInfo.shadowIntensity.x : result;
	result = (result<0.f) ? 0.f : result;
	return result;
}

/*
________________________________________________________________________________

Intersection Shader
________________________________________________________________________________
*/
__device__ float4 intersectionShader( 
	const SceneInfo& sceneInfo,
	const Primitive& primitive, 
	Material*        materials,
	char*            textures,
	const float3&    intersection,
	const bool&      back )
{
	float4 colorAtIntersection = materials[primitive.materialId.x].color;
	switch( primitive.type.x ) 
	{
	case ptCylinder:
		{
			if(materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE)
			{
				colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection );
			}
			break;
		}
	case ptEnvironment:
	case ptSphere:
   case ptEllipsoid:
		{
			if(materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE)
			{
				colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection );
			}
			break;
		}
	case ptCheckboard :
		{
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection );
			}
			else 
			{
				int x = sceneInfo.viewDistance.x + ((intersection.x - primitive.p0.x)/primitive.size.x*primitive.materialInfo.x);
				int z = sceneInfo.viewDistance.x + ((intersection.z - primitive.p0.z)/primitive.size.x*primitive.materialInfo.y);
				if(x%2==0) 
				{
					if (z%2==0) 
					{
						colorAtIntersection.x = 1.f-colorAtIntersection.x;
						colorAtIntersection.y = 1.f-colorAtIntersection.y;
						colorAtIntersection.z = 1.f-colorAtIntersection.z;
					}
				}
				else 
				{
					if (z%2!=0) 
					{
						colorAtIntersection.x = 1.f-colorAtIntersection.x;
						colorAtIntersection.y = 1.f-colorAtIntersection.y;
						colorAtIntersection.z = 1.f-colorAtIntersection.z;
					}
				}
			}
			break;
		}
	case ptXYPlane:
	case ptYZPlane:
	case ptXZPlane:
	case ptCamera:
		{
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection );
			}
			break;
		}
#if 0
	case ptTriangle:
		break;
	case ptMagicCarpet:
		{
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = magicCarpetMapping( primitive, materials, textures, intersection, levels );
			}
			break;
		}
#endif // 0
	}
	return colorAtIntersection;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ float4 primitiveShader(
	const SceneInfo&   sceneInfo,
	const PostProcessingInfo&   postProcessingInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes, 
	Primitive* primitives, const int& nbActivePrimitives,
	int* lamps, const int& nbActiveLamps,
	Material* materials, char* textures,
	float* randoms,
	const float3& origin,
	const float3& normal, 
	const int&    objectId, 
	const float3& intersection, 
	const int&    iteration,
	float4&       refractionFromColor,
	float&        shadowIntensity,
	float4&       totalBlinn)
{
	Primitive primitive = primitives[objectId];
	float4 color = materials[primitive.materialId.x].color;
	float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

	// Lamp Impact
	shadowIntensity    = 0.f;

	if( materials[primitive.materialId.x].innerIllumination.x != 0.f || materials[primitive.materialId.x].textureInfo.z == 2 )
   {
      // Wireframe returns constant color
		return color; 
   }

   if( materials[primitive.materialId.x].textureInfo.z == 1 )
	{
		// Sky box returns color with constant lightning
		return intersectionShader( 
			sceneInfo, primitive, materials, textures, 
			intersection, false );
	}

   if( sceneInfo.graphicsLevel.x>0 )
   {
	   color *= materials[primitive.materialId.x].innerIllumination.x;
	   for( int cptLamps=0; cptLamps<nbActiveLamps; cptLamps++ ) 
	   {
		   if(lamps[cptLamps] != objectId)
		   {
			   float3 center;
   		   // randomize lamp center
			   float3 size;
			   switch( primitives[lamps[cptLamps]].type.x )
			   {
			   case ptCylinder:
				   {
					   center = (primitives[lamps[cptLamps]].p0 + primitives[lamps[cptLamps]].p1)/ 2.f;
					   size.x = primitives[lamps[cptLamps]].size.y; 
					   size.y = primitives[lamps[cptLamps]].size.y; 
					   size.z = primitives[lamps[cptLamps]].size.y; 
					   break;
				   }
			   default:
				   {
					   center = primitives[lamps[cptLamps]].p0; 
					   size=primitives[lamps[cptLamps]].size; 
					   break;
				   }
			   }

			   if( sceneInfo.pathTracingIteration.x > NB_MAX_ITERATIONS )
			   {
				   int t = 3*sceneInfo.pathTracingIteration.x + int(10.f*sceneInfo.misc.y)%100;
				   center.x += materials[primitive.materialId.x].innerIllumination.y*size.x*randoms[t  ]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
				   center.y += materials[primitive.materialId.x].innerIllumination.y*size.y*randoms[t+1]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
				   center.z += materials[primitive.materialId.x].innerIllumination.y*size.z*randoms[t+2]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
			   }

            float4 shadowColor = {0.f,0.f,0.f,0.f};
			   if( sceneInfo.graphicsLevel.x>3 && materials[primitive.materialId.x].innerIllumination.x == 0.f ) 
			   {
				   shadowIntensity = processShadows(
					   sceneInfo, boundingBoxes, nbActiveBoxes,
					   primitives, materials, textures, 
					   nbActivePrimitives, center, 
					   intersection, lamps[cptLamps], iteration, shadowColor );
			   }

            // Lightning intensity decreases with distance
            if( sceneInfo.graphicsLevel.x>0 )
            {
			      Material& material = materials[primitives[lamps[cptLamps]].materialId.x];
		         float3 lightRay = center - intersection;
               float lampIntensity = 1.f-length(lightRay)/material.innerIllumination.z;
               lampIntensity = (lampIntensity<0.f) ? 0.f : lampIntensity;
               lightRay = normalize(lightRay);

			      // --------------------------------------------------------------------------------
			      // Lambert
			      // --------------------------------------------------------------------------------
	            float lambert = (postProcessingInfo.type.x==ppe_ambientOcclusion) ? 0.6f : dot(normal,lightRay);
			      lambert = (lambert<0.f) ? 0.f : lambert;
			      lambert *= (materials[primitive.materialId.x].refraction.x == 0.f) ? material.innerIllumination.x : 1.f;
			      lambert *= (1.f-shadowIntensity);
               lambert *= lampIntensity;

			      // Lighted object, not in the shades
			      lampsColor += lambert*material.color*material.innerIllumination.x - shadowColor;

			      if( sceneInfo.graphicsLevel.x>1 && shadowIntensity<sceneInfo.shadowIntensity.x )
			      {
				      // --------------------------------------------------------------------------------
				      // Blinn - Phong
				      // --------------------------------------------------------------------------------
				      float3 viewRay = normalize(intersection - origin);
				      float3 blinnDir = lightRay - viewRay;
				      float temp = sqrt(dot(blinnDir,blinnDir));
				      if (temp != 0.f ) 
				      {
					      // Specular reflection
					      blinnDir = (1.f / temp) * blinnDir;

					      float blinnTerm = dot(blinnDir,normal);
					      blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

					      blinnTerm = materials[primitive.materialId.x].specular.x * pow(blinnTerm,materials[primitive.materialId.x].specular.y);
					      totalBlinn += material.color * material.innerIllumination.x * blinnTerm;
				      }
			      }
            }
		   }

		   // Final color
		   float4 intersectionColor = 
			   intersectionShader( sceneInfo, primitive, materials, textures,
			   intersection, false );

		   color += intersectionColor*lampsColor;
		   saturateVector(color);

		   refractionFromColor = intersectionColor; // Refraction depending on color;
		   saturateVector( totalBlinn );
	   }
   }
	return color;
}

