/* 
* Raytracing Engine
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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Consts.h"
#include "Logging.h"
#include "FileMarshaller.h"

const std::string SCENEINFO = "SCENEINFO";
const std::string PRIMITIVE = "PRIMITIVE";
const std::string MATERIAL  = "MATERIAL";
const std::string TEXTURE   = "TEXTURE";
const size_t FORMAT_VERSION = 1;

FileMarshaller::FileMarshaller()
{
}


FileMarshaller::~FileMarshaller(void)
{
}

Vertex FileMarshaller::loadFromFile( GPUKernel& kernel, const std::string& filename, const Vertex& center, const float scale )
{
   LOG_INFO(1, "Loading 3D scene from " << filename );

   Vertex returnValue;
   Vertex min = {  kernel.getSceneInfo().viewDistance.x,  kernel.getSceneInfo().viewDistance.x,  kernel.getSceneInfo().viewDistance.x };
   Vertex max = { -kernel.getSceneInfo().viewDistance.x, -kernel.getSceneInfo().viewDistance.x, -kernel.getSceneInfo().viewDistance.x };

   std::ifstream myfile;
   myfile.open(filename.c_str(),std::ifstream::binary);
   if( myfile.is_open() ) 
   {
      // Format
      size_t version;
      myfile.read((char*)&version,sizeof(size_t));
      LOG_INFO(1,"Version: " << version);

      /*
      SceneInfo& sceneInfo = kernel.getSceneInfo();
      */
      SceneInfo sceneInfo; // NOT USED FOR NOW
      if(version==1)
      {
         SceneInfo1 sceneInfo1; // NOT USED FOR NOW
         myfile.read((char*)&sceneInfo1,sizeof(SceneInfo1));
      }
      else
      {
         myfile.read((char*)&sceneInfo,sizeof(SceneInfo));
      }

      // --------------------------------------------------------------------------------
      // Primitives
      // --------------------------------------------------------------------------------
      size_t nbElements(0);
      myfile.read((char*)&nbElements,sizeof(size_t));
      LOG_INFO(1,"Loading " << nbElements << " primitives...");

      size_t position=myfile.tellg();

      for( size_t i(0); i<nbElements; ++i)
      {
         CPUPrimitive primitive;
         myfile.read((char*)&primitive,sizeof(CPUPrimitive));

         int n = kernel.addPrimitive( static_cast<PrimitiveType>(primitive.type) );
         kernel.setPrimitive(
            n, 
            center.x+primitive.p0.x, center.y+primitive.p0.y, center.z+primitive.p0.z,
            center.x+primitive.p1.x, center.y+primitive.p1.y, center.z+primitive.p1.z,
            center.x+primitive.p2.x, center.y+primitive.p2.y, center.z+primitive.p2.z,
            primitive.size.x, primitive.size.y, primitive.size.z,
            primitive.materialId );
         kernel.setPrimitiveBellongsToModel(n,true);
         kernel.setPrimitiveNormals( n, primitive.n0, primitive.n1, primitive.n2 );
         kernel.setPrimitiveTextureCoordinates( n, primitive.vt0, primitive.vt1, primitive.vt2 );

         Vertex pmin, pmax;
         pmin.x = std::min(std::min( primitive.p0.x, primitive.p1.x ), primitive.p2.x );
         pmin.y = std::min(std::min( primitive.p0.y, primitive.p1.y ), primitive.p2.y );
         pmin.z = std::min(std::min( primitive.p0.z, primitive.p1.z ), primitive.p2.z );
         pmax.x = std::max(std::max( primitive.p0.x, primitive.p1.x ), primitive.p2.x );
         pmax.y = std::max(std::max( primitive.p0.y, primitive.p1.y ), primitive.p2.y );
         pmax.z = std::max(std::max( primitive.p0.z, primitive.p1.z ), primitive.p2.z );

         min.x = ( pmin.x < min.x ) ? pmin.x : min.x;
         min.y = ( pmin.y < min.y ) ? pmin.y : min.y;
         min.z = ( pmin.z < min.z ) ? pmin.z : min.z;

         max.x = ( pmax.x > max.x ) ? pmax.x : max.x;
         max.y = ( pmax.y > max.y ) ? pmax.y : max.y;
         max.z = ( pmax.z > max.z ) ? pmax.z : max.z;
      }

      // --------------------------------------------------------------------------------
      // Textures
      // --------------------------------------------------------------------------------
      size_t nbTextures(0);
      myfile.read((char*)&nbTextures,sizeof(size_t));
      LOG_INFO(1,"Loading " << nbTextures << " textures...");

      int nbActiveTextures=kernel.getNbActiveTextures();
      for( unsigned int i(0); i<nbTextures; ++i)
      {
         size_t id;
         myfile.read((char*)&id,sizeof(size_t));
            
         TextureInformation texInfo;
         myfile.read((char*)&texInfo,sizeof(TextureInformation));
         LOG_INFO(1,"Texture with id " << id << " and size: " << texInfo.size.x << "x" << texInfo.size.y << "x" << texInfo.size.z << " loaded into slot " << nbActiveTextures+i);

         size_t imageSize=texInfo.size.x*texInfo.size.y*texInfo.size.z;
         texInfo.buffer=new BitmapBuffer[imageSize];
         myfile.read((char*)texInfo.buffer,imageSize);

         kernel.setTexture(static_cast<const int>(nbActiveTextures+i),texInfo);
         delete [] texInfo.buffer;
      }

      // --------------------------------------------------------------------------------
      // Materials
      // --------------------------------------------------------------------------------
      size_t nbMaterials(0);
      myfile.read((char*)&nbMaterials,sizeof(size_t));
      LOG_INFO(1,"Loading " << nbMaterials << " materials...");

      for( unsigned int i(0); i<nbMaterials; ++i)
      {
         size_t id;
         Material material;
         myfile.read((char*)&id,sizeof(size_t));
         myfile.read((char*)&material,sizeof(Material));
         if(material.textureIds.x!=TEXTURE_NONE) material.textureIds.x += nbActiveTextures;
         if(material.textureIds.y!=TEXTURE_NONE) material.textureIds.y += nbActiveTextures;
         if(material.textureIds.z!=TEXTURE_NONE) material.textureIds.z += nbActiveTextures;
         if(material.textureIds.w!=TEXTURE_NONE) material.textureIds.w += nbActiveTextures;
         if(material.advancedTextureIds.x!=TEXTURE_NONE) material.advancedTextureIds.x += nbActiveTextures;
         if(material.advancedTextureIds.y!=TEXTURE_NONE) material.advancedTextureIds.y += nbActiveTextures;
         if(material.advancedTextureIds.z!=TEXTURE_NONE) material.advancedTextureIds.z += nbActiveTextures;
         if(material.advancedTextureIds.w!=TEXTURE_NONE) material.advancedTextureIds.w += nbActiveTextures;
         LOG_INFO(1,"Loading material " << id << " (" << 
            material.textureIds.x << "," << material.textureIds.y << "," << material.textureIds.z << "," << material.textureIds.w << 
            material.advancedTextureIds.x << "," << material.advancedTextureIds.y << "," << material.advancedTextureIds.z << "," << material.advancedTextureIds.w << 
            ")");
         kernel.setMaterial(static_cast<unsigned int>(id),material);
      }
   }
   myfile.close();
   LOG_INFO(1,"File " << filename << " successfully loaded!");

   // Object size
   returnValue.x = fabs( max.x - min.x );
   returnValue.y = fabs( max.y - min.y );
   returnValue.z = fabs( max.z - min.z );

   // Resize to fit required size
   float ratio = scale / returnValue.y;
   kernel.scalePrimitives( ratio, 0, NB_MAX_BOXES );

   LOG_INFO(1, "Object size: " << returnValue.x << ", " << returnValue.y << ", " << returnValue.z );
   return returnValue;
}

void FileMarshaller::saveToFile( GPUKernel& kernel, const std::string& filename)
{
   int frame(0);
   LOG_INFO(1, "Saving 3D scene to " << filename );
   std::ofstream myfile;
   myfile.open(filename.c_str(),std::ifstream::binary);
   if( myfile.is_open() ) 
   {
      // Format version
      size_t version=FORMAT_VERSION;
      myfile.write((char*)&version,sizeof(size_t));

      std::map<size_t,Material*> materials;

      // Scene
      SceneInfo& sceneInfo = kernel.getSceneInfo();
      myfile.write((char*)&sceneInfo,sizeof(SceneInfo));

      // Primitives

      // Count primitives belonging to the model
      size_t nbTotalPrimitives = kernel.getNbActivePrimitives();
      size_t nbPrimitives(0);
      for( int i(0); i<nbTotalPrimitives; ++i )
      {
         if(kernel.getPrimitive(i)->belongsToModel) ++nbPrimitives;
      }

      myfile.write((char*)&nbPrimitives,sizeof(size_t));
      LOG_INFO(1,"Saving " << nbPrimitives << " primitives");

      // Identify used materials
      std::map<int,int> materialIndexMapping;
      for( int i(0); i<nbPrimitives; ++i )
      {
         CPUPrimitive* primitive = kernel.getPrimitive(i);
         myfile.write((char*)primitive,sizeof(CPUPrimitive));
         materials[primitive->materialId] = kernel.getMaterial(primitive->materialId);
      }

      // Determine textures in use
      std::map<size_t,TextureInformation> textures;
      std::map<size_t,Material*>::const_iterator ittiu(materials.begin());
      while( ittiu!=materials.end() )
      {
         Material* material = (*ittiu).second;
         if( material->textureIds.x!=TEXTURE_NONE ) textures[material->textureIds.x] = kernel.getTextureInformation(material->textureIds.x);
         if( material->textureIds.y!=TEXTURE_NONE ) textures[material->textureIds.y] = kernel.getTextureInformation(material->textureIds.y);
         if( material->textureIds.z!=TEXTURE_NONE ) textures[material->textureIds.z] = kernel.getTextureInformation(material->textureIds.z);
         if( material->textureIds.w!=TEXTURE_NONE ) textures[material->textureIds.w] = kernel.getTextureInformation(material->textureIds.w);
         if( material->advancedTextureIds.x!=TEXTURE_NONE ) textures[material->advancedTextureIds.x] = kernel.getTextureInformation(material->advancedTextureIds.x);
         if( material->advancedTextureIds.y!=TEXTURE_NONE ) textures[material->advancedTextureIds.y] = kernel.getTextureInformation(material->advancedTextureIds.y);
         if( material->advancedTextureIds.z!=TEXTURE_NONE ) textures[material->advancedTextureIds.z] = kernel.getTextureInformation(material->advancedTextureIds.z);
         if( material->advancedTextureIds.w!=TEXTURE_NONE ) textures[material->advancedTextureIds.w] = kernel.getTextureInformation(material->advancedTextureIds.w);
         ++ittiu;
      }

      // Write Textures
      size_t nbTextures = textures.size();
      myfile.write((char*)&nbTextures,sizeof(size_t));
      LOG_INFO(1,"Saving " << nbTextures << " textures");

      std::map<size_t,int> idMapping;
      std::map<size_t,TextureInformation>::const_iterator itt=textures.begin();
      size_t index(0);
      idMapping[TEXTURE_NONE]=TEXTURE_NONE;
      while( itt!=textures.end() )
      {
         TextureInformation texInfo=(*itt).second;
         BitmapBuffer* savedBuffer=texInfo.buffer;
         texInfo.buffer=0;
         texInfo.offset=0;
         size_t id=(*itt).first;
         idMapping[id]=static_cast<int>(index);
         LOG_INFO(1,"Texture " << id << ": " << texInfo.size.x << "x" << texInfo.size.y << "x" << texInfo.size.z << " saved with id " << index);
         myfile.write((char*)(&index),sizeof(size_t));
         myfile.write((char*)(&texInfo),sizeof(TextureInformation));
         myfile.write((char*)(savedBuffer),texInfo.size.x*texInfo.size.y*texInfo.size.z);
         ++index;
         ++itt;
      }

      // Write Materials
      size_t nbMaterials=materials.size();
      myfile.write((char*)&nbMaterials,sizeof(size_t));
      LOG_INFO(1,"Saving " << nbMaterials << " materials");

      std::map<size_t,Material*>::const_iterator itm(materials.begin());
      while( itm!=materials.end() )
      {
         Material material=*(*itm).second;
         material.textureIds.x = idMapping[material.textureIds.x];
         material.textureIds.y = idMapping[material.textureIds.y];
         material.textureIds.z = idMapping[material.textureIds.z];
         material.textureIds.w = idMapping[material.textureIds.w];
         material.advancedTextureIds.x = idMapping[material.advancedTextureIds.x];
         material.advancedTextureIds.y = idMapping[material.advancedTextureIds.y];
         material.advancedTextureIds.z = idMapping[material.advancedTextureIds.z];
         material.advancedTextureIds.w = idMapping[material.advancedTextureIds.w];
         myfile.write((char*)&(*itm).first,sizeof(size_t));
         myfile.write((char*)&material,sizeof(Material));
         LOG_INFO(1,"Saving material " << (*itm).first << " (" << 
            material.textureIds.x << "," << material.textureIds.y << "," << material.textureIds.z << "," << material.textureIds.w << "," <<
            material.advancedTextureIds.x << "," << material.advancedTextureIds.y << "," << material.advancedTextureIds.z << "," << material.advancedTextureIds.w << ")"
            );
         ++itm;
      }

      myfile.close();
   }
}
