/* 
* Raytracing Engine
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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>

#include "FileMarshaller.h"

const std::string SCENEINFO = "SCENEINFO";
const std::string PRIMITIVE = "PRIMITIVE";
const std::string MATERIAL  = "MATERIAL";

FileMarshaller::FileMarshaller(GPUKernel* kernel)
   : m_gpuKernel( kernel )
{
}


FileMarshaller::~FileMarshaller(void)
{
}

void FileMarshaller::readSceneInfo( const std::string& line )
{
   SceneInfo sceneInfo;
   std::string value;
   size_t i( strlen(SCENEINFO.c_str())+1 );
   size_t c(0);
   while( i<line.length() )
   {
      if( line[i] == ';' ) 
      {
         switch(c)
         {
            case  0: sceneInfo.width.x  = static_cast<int>(atoi( value.c_str() )); break;
            case  1: sceneInfo.height.x = static_cast<int>(atoi( value.c_str() )); break;
            case  2: sceneInfo.shadowsEnabled.x = static_cast<int>(atoi( value.c_str() )); break;
            case  3: sceneInfo.nbRayIterations.x = static_cast<int>(atoi( value.c_str() )); break;
            case  4: sceneInfo.transparentColor.x = static_cast<float>(atof( value.c_str() )); break;
            case  5: sceneInfo.viewDistance.x = static_cast<float>(atof( value.c_str() )); break;
            case  6: sceneInfo.shadowIntensity.x = static_cast<float>(atof( value.c_str() )); break;
            case  7: sceneInfo.width3DVision.x = static_cast<float>(atof( value.c_str() )); break;
            case  8: sceneInfo.backgroundColor.x = static_cast<float>(atof( value.c_str() )); break;
            case  9: sceneInfo.backgroundColor.y = static_cast<float>(atof( value.c_str() )); break;
            case 10: sceneInfo.backgroundColor.z = static_cast<float>(atof( value.c_str() )); break;
            case 11: sceneInfo.backgroundColor.w = static_cast<float>(atof( value.c_str() )); break;
            case 12: sceneInfo.supportFor3DVision.x = static_cast<int>(atoi( value.c_str() )); break;
            case 13: sceneInfo.renderBoxes.x = static_cast<int>(atoi( value.c_str() )); break;
            case 14: sceneInfo.pathTracingIteration.x = static_cast<int>(atoi( value.c_str() )); break;
            case 15: sceneInfo.maxPathTracingIterations.x = static_cast<int>(atoi( value.c_str() )); break;
            case 16: sceneInfo.misc.x = static_cast<int>(atoi( value.c_str() )); break;
            case 17: sceneInfo.misc.y = static_cast<int>(atoi( value.c_str() )); break;
         }
         value = "";
         ++i;
         c++;
      }
      value += line[i];
      ++i;
   }

   m_gpuKernel->setSceneInfo( sceneInfo );
	m_gpuKernel->initBuffers();
}

void FileMarshaller::readPrimitive( const std::string& line )
{
   Primitive primitive;
   std::string value;
   size_t i( strlen(PRIMITIVE.c_str())+1 );
   int c(0);
   int boxId(0);

   while( i<line.length() )
   {
      if( line[i] == ';' ) 
      {
         switch(c)
         {
            case  0: boxId = static_cast<int>(atoi( value.c_str() )); break;
            case  1: primitive.type.x   = static_cast<int>(atoi( value.c_str() )); break;

            case  2: primitive.p0.x = static_cast<float>(atof( value.c_str() )); break;
            case  3: primitive.p0.y = static_cast<float>(atof( value.c_str() )); break;
            case  4: primitive.p0.z = static_cast<float>(atof( value.c_str() )); break;
            case  5: primitive.p0.w = static_cast<float>(atof( value.c_str() )); break;

            case  6: primitive.p1.x = static_cast<float>(atof( value.c_str() )); break;
            case  7: primitive.p1.y = static_cast<float>(atof( value.c_str() )); break;
            case  8: primitive.p1.z = static_cast<float>(atof( value.c_str() )); break;
            case  9: primitive.p1.w = static_cast<float>(atof( value.c_str() )); break;

            case 10: primitive.size.x = static_cast<float>(atof( value.c_str() )); break;
            case 11: primitive.size.y = static_cast<float>(atof( value.c_str() )); break;
            case 12: primitive.size.z = static_cast<float>(atof( value.c_str() )); break;
            case 13: primitive.size.w = static_cast<float>(atof( value.c_str() )); break;

            case 14: primitive.n0.x = static_cast<float>(atof( value.c_str() )); break;
            case 15: primitive.n0.y = static_cast<float>(atof( value.c_str() )); break;
            case 16: primitive.n0.z = static_cast<float>(atof( value.c_str() )); break;
            case 17: primitive.n0.w = static_cast<float>(atof( value.c_str() )); break;

            case 18: primitive.materialId.x = static_cast<int>(atoi( value.c_str() )); break;
            case 19: primitive.materialInfo.x = static_cast<float>(atof( value.c_str() )); break;
         }
         value = "";
         ++i;
         c++;
      }
      value += line[i];
      ++i;
   }

   if( value.length() != 0 )
   {
      primitive.materialInfo.y = static_cast<float>(atof( value.c_str() ));
   }

   int n = m_gpuKernel->addPrimitive( static_cast<PrimitiveType>(primitive.type.x) );
   m_gpuKernel->setPrimitive(
      n, boxId,
      primitive.p0.x, primitive.p0.y, primitive.p0.z,
      primitive.p1.x, primitive.p1.y, primitive.p1.z,
      primitive.p2.x, primitive.p2.y, primitive.p2.z,
      primitive.size.x, primitive.size.y, primitive.size.z,
      primitive.materialId.x, 
      primitive.materialInfo.x, 
      primitive.materialInfo.y );
}

void FileMarshaller::readMaterial( const std::string& line )
{
   Material material;
   std::string value;
   size_t i( strlen(MATERIAL.c_str())+1 );
   size_t c(0);
   size_t boxId(0);

   while( i<line.length() )
   {
      if( line[i] == ';' ) 
      {
         switch(c)
         {
            case  0: material.color.x   = static_cast<float>(atof( value.c_str() )); break;
            case  1: material.color.y   = static_cast<float>(atof( value.c_str() )); break;
            case  2: material.color.z   = static_cast<float>(atof( value.c_str() )); break;
            case  3: material.color.w   = static_cast<float>(atof( value.c_str() )); break;
            case  4: material.innerIllumination.x = static_cast<float>(atof( value.c_str() )); break;
            case  5: material.innerIllumination.y = static_cast<float>(atof( value.c_str() )); break;
            case  6: material.innerIllumination.z = static_cast<float>(atof( value.c_str() )); break;
            case  7: material.innerIllumination.w = static_cast<float>(atof( value.c_str() )); break;
            case  8: material.reflection.x        = static_cast<float>(atof( value.c_str() )); break;
            case  9: material.refraction.x        = static_cast<float>(atof( value.c_str() )); break;
            case 10: material.specular.x          = static_cast<float>(atof( value.c_str() )); break;
            case 11: material.specular.y          = static_cast<float>(atof( value.c_str() )); break;
            case 12: material.specular.z          = static_cast<float>(atof( value.c_str() )); break;
            case 13: material.specular.w          = static_cast<float>(atof( value.c_str() )); break;
            case 14: material.textureInfo.x       = static_cast<int>(atoi( value.c_str() )); break;
            case 15: material.textureInfo.y       = static_cast<int>(atoi( value.c_str() )); break;
            case 16: material.textureInfo.z       = static_cast<int>(atoi( value.c_str() )); break;
            case 17: material.textureInfo.w       = static_cast<int>(atoi( value.c_str() )); break;
            case 18: material.transparency.x      = static_cast<float>(atof( value.c_str() )); break;
         }
         value = "";
         ++i;
         c++;
      }
      value += line[i];
      ++i;
   }

   if( value.length() != 0 )
   {
      material.fastTransparency.x = static_cast<int>(atoi( value.c_str() ));
   }

   int n = m_gpuKernel->addMaterial();
   m_gpuKernel->setMaterial(
      n,
      material.color.x, material.color.y, material.color.z, material.color.w,
      material.reflection.x, material.refraction.x,
      (material.textureInfo.x==1), (material.textureInfo.z==1), material.textureInfo.w, // TODO!!!
      material.transparency.x, material.textureInfo.y,
      material.specular.x, material.specular.y, material.specular.z,
      material.innerIllumination.x,
      (material.fastTransparency.x==1));
      
}

void FileMarshaller::loadFromFile( const std::string& filename)
{
   std::ifstream myfile;
   myfile.open(filename.c_str());
   if( myfile.is_open() ) 
   {
      while( !myfile.eof() ) 
      {
         std::string line;
         std::getline(myfile, line);
         if( line.find(SCENEINFO) == 0 )
         {
            readSceneInfo( line );
         }
         else if( line.find(MATERIAL) == 0 )
         {
            readMaterial( line );
         }
         else if( line.find(PRIMITIVE) == 0 )
         {
            readPrimitive( line );
         }
      }
   }
   myfile.close();
   m_gpuKernel->compactBoxes();
}

void FileMarshaller::saveToFile( const std::string& filename)
{
   std::ofstream myfile;
   myfile.open(filename.c_str());
   if( myfile.is_open() ) 
   {
      SceneInfo sceneInfo = m_gpuKernel->getSceneInfo();
      // Scene
      myfile << SCENEINFO << ";" <<
         sceneInfo.width.x << ";" <<
         sceneInfo.height.x << ";" <<
         sceneInfo.shadowsEnabled.x << ";" <<
         sceneInfo.nbRayIterations.x << ";" <<
         sceneInfo.transparentColor.x << ";" <<
         sceneInfo.viewDistance.x << ";" <<
         sceneInfo.shadowIntensity.x << ";" <<
         sceneInfo.width3DVision.x << ";" <<
         sceneInfo.backgroundColor.x << ";" <<
         sceneInfo.backgroundColor.y << ";" <<
         sceneInfo.backgroundColor.z << ";" <<
         sceneInfo.backgroundColor.w << ";" <<
         sceneInfo.supportFor3DVision.x << ";" <<
         sceneInfo.renderBoxes.x << ";" <<
         sceneInfo.pathTracingIteration.x << ";" <<
         sceneInfo.maxPathTracingIterations.x << ";" <<
         sceneInfo.misc.x << ";" <<
         sceneInfo.misc.y << std::endl;

      // Materials
      for( int i(0); i<=m_gpuKernel->getNbActiveMaterials(); ++i )
      {
         Material* material = m_gpuKernel->getMaterial(i);
         myfile << "MATERIAL;" <<  
            material->color.x << ";" <<
            material->color.y << ";" <<
            material->color.z << ";" <<
            material->color.w << ";" <<
            material->innerIllumination.x << ";" <<
            material->innerIllumination.y << ";" <<
            material->innerIllumination.z << ";" <<
            material->innerIllumination.w << ";" <<
            material->reflection.x << ";" <<
            material->refraction.x << ";" <<
            material->specular.x << ";" <<
            material->specular.y << ";" <<
            material->specular.z << ";" <<
            material->specular.w << ";" <<
            material->textureInfo.x << ";" <<
            material->textureInfo.y << ";" <<
            material->textureInfo.z << ";" <<
            material->textureInfo.w << ";" <<
            material->transparency.x << ";" <<
            material->fastTransparency.x << std::endl;
      }
      // Primitives
      for( int b(0); b<=m_gpuKernel->getNbActiveBoxes(); ++b )
      {
         BoundingBox& box = m_gpuKernel->getBoundingBox(b);
         
         for( int i(0); i<box.nbPrimitives.x; ++i )
         {
            Primitive* primitive = m_gpuKernel->getPrimitive( box.startIndex.x+i );
            myfile << "PRIMITIVE;" <<  
               b << ";" <<
               primitive->type.x << ";" <<
               primitive->p0.x << ";" <<
               primitive->p0.y << ";" <<
               primitive->p0.z << ";" <<
               primitive->p0.w << ";" <<
               primitive->p1.x << ";" <<
               primitive->p1.y << ";" <<
               primitive->p1.z << ";" <<
               primitive->p1.w << ";" <<
               primitive->size.x << ";" <<
               primitive->size.y << ";" <<
               primitive->size.z << ";" <<
               primitive->size.w << ";" <<
               primitive->n0.x << ";" <<
               primitive->n0.y << ";" <<
               primitive->n0.z << ";" <<
               primitive->n0.w << ";" <<
               primitive->materialId.x << ";" <<
               primitive->materialInfo.x << ";" <<
               primitive->materialInfo.y << std::endl;
         }
      }
   }
   myfile.close();
}
