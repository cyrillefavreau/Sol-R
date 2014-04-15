/* 
* Copyright (C) 2011-2014 Cyrille Favreau <cyrille_favreau@hotmail.com>
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
#include <vector_functions.h>

/*
________________________________________________________________________________

Convert Vertex into OpenGL RGB color
________________________________________________________________________________
*/
__device__ __INLINE__ void makeColor(
	const SceneInfo& sceneInfo,
	float4&          color,
	BitmapBuffer*    bitmap,
	int              index)
{
   int mdc_index = index*gColorDepth; 
	color.x = (color.x>1.f) ? 1.f : color.x;
	color.y = (color.y>1.f) ? 1.f : color.y; 
	color.z = (color.z>1.f) ? 1.f : color.z;
	color.x = (color.x<0.f) ? 0.f : color.x;
	color.y = (color.y<0.f) ? 0.f : color.y; 
	color.z = (color.z<0.f) ? 0.f : color.z;

   switch( sceneInfo.misc.x )
   {
      case otOpenGL: 
	   {
		   // OpenGL
		   bitmap[mdc_index  ] = (BitmapBuffer)(color.x*255.f); // Red
      	bitmap[mdc_index+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (BitmapBuffer)(color.z*255.f); // Blue
         break;
	   }
      case otDelphi: 
	   {
		   // Delphi
		   bitmap[mdc_index  ] = (BitmapBuffer)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (BitmapBuffer)(color.x*255.f); // Red
         break;
	   }
      case otJPEG: 
	   {
         mdc_index = (sceneInfo.width.x*sceneInfo.height.x-index)*gColorDepth; 
		   // JPEG
		   bitmap[mdc_index+2] = (BitmapBuffer)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[mdc_index  ] = (BitmapBuffer)(color.x*255.f); // Red
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
which origin is the considered 3D Vertex and which direction is defined by the 
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
__device__ __INLINE__ float processShadows(
	const SceneInfo& sceneInfo,
	BoundingBox*  boudingBoxes, const int& nbActiveBoxes,
	Primitive*    primitives,
	Material*     materials,
	BitmapBuffer* textures,
	const int&    nbPrimitives, 
	const Vertex& lampCenter, 
	const Vertex& origin, 
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
				Vertex intersection = {0.f,0.f,0.f};
				Vertex normal       = {0.f,0.f,0.f};
            Vertex areas        = {0.f,0.f,0.f};
				float  shadowIntensity = 0.f;

				Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
            if( primitive.index.x!=objectId && materials[primitive.materialId.x].attributes.x==0)
				{

					bool back;
					bool hit = false;
#ifdef EXTENDED_GEOMETRY
					switch(primitive.type.x)
					{
					case ptSphere   : hit=sphereIntersection   ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); break;
               case ptEllipsoid: hit=ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); break;
					case ptCylinder :	hit=cylinderIntersection ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); break;
					case ptTriangle :	hit=triangleIntersection ( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, back, true ); break;
					case ptCamera   : hit=false; break;
					default         : hit=planeIntersection    ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false ); break;
					}
#else
               hit = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, back, true );
#endif

#ifdef EXTENDED_FEATURES
               if( hit && sceneInfo.transparentColor.x!=0.f)
               {
		            float4 closestColor = intersectionShader( 
			            sceneInfo, primitive, materials, textures, 
			            intersection, areas );
                  hit = ((closestColor.x+closestColor.y+closestColor.z)>sceneInfo.transparentColor.x);
               }
#endif // EXTENDED_FEATURES

               if( hit )
					{
						Vertex O_I = intersection-r.origin;
						Vertex O_L = r.direction;
						float l = length(O_I);
						if( l>EPSILON && l<length(O_L) )
						{
                     float ratio = shadowIntensity*sceneInfo.shadowIntensity.x;
                     if( materials[primitive.materialId.x].transparency.x != 0.f )
                     {
                        O_L=normalize(O_L);
                        float a=fabs(dot(O_L,normal));
                        float r = (materials[primitive.materialId.x].transparency.x == 0.f ) ? 1.f : (1.f-/*0.8f**/materials[primitive.materialId.x].transparency.x);
                        ratio *= r*a;
                        // Shadow color
                        color.x  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.x);
                        color.y  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.y);
                        color.z  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.z);
                     }
                     result += ratio;
                  }
                  it++;
					}
				}
				cptPrimitives++;
			}
   		cptBoxes++;
		}
      else
      {
         cptBoxes+=box.indexForNextBox.x;
      }
	}
	result = (result>sceneInfo.shadowIntensity.x) ? sceneInfo.shadowIntensity.x : result;
	result = (result<0.f) ? 0.f : result;
	return result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ __INLINE__ float4 primitiveShader(
   const int& index,
	const SceneInfo&   sceneInfo,
	const PostProcessingInfo&   postProcessingInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes, 
	Primitive* primitives, const int& nbActivePrimitives,
	LightInformation* lightInformation, const int& lightInformationSize, const int& nbActiveLamps,
	Material* materials, BitmapBuffer* textures,
	RandomBuffer* randoms,
   const Vertex& origin,
	Vertex&       normal, 
	const int&    objectId, 
	Vertex&       intersection,
   const Vertex& areas,
   float4&       closestColor,
	const int&    iteration,
	float4&       refractionFromColor,
	float&        shadowIntensity,
	float4&       totalBlinn,
   Vertex&       attributes)
{
	Primitive& primitive = primitives[objectId];
   Material& material = materials[primitive.materialId.x];
	float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

	// Lamp Impact
	shadowIntensity    = 0.f;

   // Bump
   Vertex bumpNormal={0.f,0.f,0.f};

   // Specular
   Vertex specular;
   specular.x=material.specular.x;
   specular.y=material.specular.y;
   specular.z=material.specular.z;

   // Intersection color
   float4 intersectionColor = intersectionShader( sceneInfo, primitive, materials, textures, intersection, areas, bumpNormal, specular, attributes );
   normal += bumpNormal;
   normal = normalize(normal);

	if( material.innerIllumination.x!=0.f || material.attributes.z!=0 )
   {
      // Wireframe returns constant color
		return intersectionColor; 
   }

   if( sceneInfo.graphicsLevel.x>0 )
   {
		// Final color
#ifdef EXTENDED_FEATURES
      // TODO: Bump effect
      if( materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE)
      {
         normal.x = normal.x*0.7f+intersectionColor.x*0.3f;
         normal.y = normal.y*0.7f+intersectionColor.y*0.3f;
         normal.z = normal.z*0.7f+intersectionColor.z*0.3f;
      }
#endif // EXTENDED_FEATURES

	   closestColor *= material.innerIllumination.x;
	   for( int cpt=0; cpt<lightInformationSize; ++cpt ) 
	   {
         int cptLamp = cpt;
         if(lightInformation[cptLamp].attribute.x != primitive.index.x)
		   {
			   Vertex center;
   		   // randomize lamp center
            center.x = lightInformation[cptLamp].location.x;
            center.y = lightInformation[cptLamp].location.y;
            center.z = lightInformation[cptLamp].location.z;

            int t = (index*3+sceneInfo.misc.y)%(sceneInfo.width.x*sceneInfo.height.x);
            Material& m=materials[lightInformation[cptLamp].attribute.y];
            if( lightInformation[cptLamp].attribute.x>=0 &&
                lightInformation[cptLamp].attribute.x<nbActivePrimitives)
            {
               t = t%(sceneInfo.width.x*sceneInfo.height.x-3);
               float a=(sceneInfo.pathTracingIteration.x<sceneInfo.maxPathTracingIterations.x) ? sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x) : 1.f;
               center.x += m.innerIllumination.y*randoms[t  ]*a;
				   center.y += m.innerIllumination.y*randoms[t+1]*a;
				   center.z += m.innerIllumination.y*randoms[t+2]*a;
            }

		      Vertex lightRay = center - intersection;
            float lightRayLength=length(lightRay);
            if( lightRayLength<m.innerIllumination.z )
            {
               float4 shadowColor = {0.f,0.f,0.f,0.f};
               if( sceneInfo.graphicsLevel.x>3 && 
                   iteration<4 && // No need to process shadows after 4 generations of rays... cannot be seen anyway.
                   material.innerIllumination.x==0.f ) 
			      {
				      shadowIntensity = processShadows(
					      sceneInfo, boundingBoxes, nbActiveBoxes,
					      primitives, materials, textures, 
					      nbActivePrimitives, center, 
                     intersection, lightInformation[cptLamp].attribute.x, iteration, shadowColor );
			      }

               if( sceneInfo.graphicsLevel.x>0 )
               {
//#ifdef PHOTON_ENERGY
                  float photonEnergy = sqrt(lightRayLength/m.innerIllumination.z);
                  photonEnergy = (photonEnergy>1.f) ? 1.f : photonEnergy;
                  photonEnergy = (photonEnergy<0.f) ? 0.f : photonEnergy;
//#endif // PHOTON_ENERGY

                  lightRay = normalize(lightRay);
			         // --------------------------------------------------------------------------------
			         // Lambert
			         // --------------------------------------------------------------------------------
	               float lambert = dot(normal,lightRay); // (postProcessingInfo.type.x==ppe_ambientOcclusion) ? 0.6f : dot(normal,lightRay);
                  // Transparent materials are lighted on both sides but the amount of light received by the "dark side" 
                  // depends on the transparency rate.
                  lambert *= (lambert<0.f) ? -materials[primitive.materialId.x].transparency.x : 1.f;

                  if( lightInformation[cptLamp].attribute.y != MATERIAL_NONE )
                  {
                     Material& m=materials[lightInformation[cptLamp].attribute.y];
                     lambert *= m.innerIllumination.x; // Lamp illumination
                  }
                  else
                  {
                     lambert *= lightInformation[cptLamp].color.w;
                  }

                  lambert *= (1.f+randoms[t]*material.innerIllumination.w); // Randomize lamp intensity depending on material noise, for more realistic rendering
			         lambert *= (1.f-shadowIntensity);
                  lambert += sceneInfo.backgroundColor.w;
                  lambert *= (1.f-photonEnergy);

                  // Lighted object, not in the shades
                  lampsColor += lambert*lightInformation[cptLamp].color - shadowColor;

			         if( sceneInfo.graphicsLevel.x>1 && shadowIntensity<sceneInfo.shadowIntensity.x )
			         {
				         // --------------------------------------------------------------------------------
				         // Blinn - Phong
				         // --------------------------------------------------------------------------------
				         Vertex viewRay = normalize(intersection - origin);
				         Vertex blinnDir = lightRay - viewRay;
				         float temp = sqrt(dot(blinnDir,blinnDir));
				         if (temp != 0.f ) 
				         {
					         // Specular reflection
					         blinnDir = (1.f / temp) * blinnDir;

					         float blinnTerm = dot(blinnDir,normal);
					         blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

					         blinnTerm = specular.x * pow(blinnTerm,specular.y);
                        blinnTerm *= (1.f-material.transparency.x);
                        blinnTerm *= (1.f-photonEnergy);
					         totalBlinn += lightInformation[cptLamp].color * lightInformation[cptLamp].color.w * blinnTerm;
                     
                        // Get transparency from specular map
                        totalBlinn.w = specular.z;
				         }
			         }
               }
		      }
         }

         // Light impact on material
		   closestColor += intersectionColor*lampsColor;

         // Saturate color
		   saturateVector(closestColor);

		   refractionFromColor = intersectionColor; // Refraction depending on color;
		   saturateVector( totalBlinn );
	   }
   }
   else
   {
      closestColor = intersectionColor;
   }
	return closestColor;
}

