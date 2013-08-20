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

#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>

#include "Logging.h"

#include "Consts.h"
#include "OBJReader.h"

OBJReader::OBJReader(void)
{
}

OBJReader::~OBJReader(void)
{
}

float3 readfloat3( const std::string& value )
{
   float3 returnValue = {0.f,0.f,0.f};
   int item(0);
   std::string tmp;
   for( int i(0); i<value.length(); ++i)
   {
      if( value[i] != ' ') tmp += value[i];

      if( value[i] == ' ' || i==(value.length()-1) )
      {
         switch( item )
         {
            case 0: returnValue.x = static_cast<float>(atof(tmp.c_str())); break;
            case 1: returnValue.y = static_cast<float>(atof(tmp.c_str())); break;
            case 2: returnValue.z = static_cast<float>(atof(tmp.c_str())); break;
         }
         ++item;
         tmp = "";
      }
   }
   return returnValue;
}

int4 readVertex( const std::string& face )
{
   int4 returnValue = {0,0,0,0};
   int item(0);
   std::string value;
   for( int i(0); i<face.length(); ++i)
   {
      if( face[i] != ' ') value += face[i];

      if( face[i] == ' ' || i==(face.length()-1) )
      {
         switch( item )
         {
            case 0: returnValue.x = atoi(value.c_str()); break;
            case 1: returnValue.y = atoi(value.c_str()); break;
            case 2: returnValue.z = atoi(value.c_str()); break;
         }
         ++item;
         value = "";
      }
   }
   return returnValue;
}

int4 readFace( const std::string& face )
{
   int4 returnValue = {0,0,0,0};
   int item(0);
   std::string value;
   for( int i(0); i<face.length(); ++i)
   {
      if( face[i] != '/') value += face[i];

      if( face[i] == '/' /*|| i==(face.length()-1)*/ )
      {
         switch( item )
         {
            case 0: returnValue.x = (atoi(value.c_str())); break;
            case 1: returnValue.y = (atoi(value.c_str())); break;
            case 2: returnValue.z = (atoi(value.c_str())); break;
         }
         ++item;
         value = "";
      }
   }

   if( value.length() != 0 )
   {
      switch( item )
      {
         case 0: returnValue.x = (atoi(value.c_str())); break;
         case 1: returnValue.y = (atoi(value.c_str())); break;
         case 2: returnValue.z = (atoi(value.c_str())); break;
      }
   }

   //std::cout << returnValue.x <<"," << returnValue.y << "," << returnValue.z << std::endl;
   return returnValue;
}

unsigned int OBJReader::loadMaterialsFromFile(
   const std::string& filename,
   std::map<std::string,MaterialMTL>& materials,
   GPUKernel& kernel,
   int materialId)
{
   std::string materialsFilename(filename);
   materialsFilename += ".mtl";

   std::string id("");
   std::ifstream file(materialsFilename);
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         line.erase( std::remove(line.begin(), line.end(), '\r'), line.end());

         // remove spaces
         while( line.length() != 0 && line[0] < 32 )
         {
            line = line.substr(1);
         }

         if( line.find("newmtl") == 0 )
         {
            if( id.length() != 0 )
            {
               MaterialMTL& m = materials[id];
               kernel.setMaterial(
                  m.index,
                  m.Kd.x,m.Kd.y,m.Kd.z,
                  0.f,m.reflection,m.refraction, false, false, 0,
                  m.transparency, m.textureId,
                  m.Ks.x, 200.f*m.Ks.y, m.Ks.z,
                  0.f, 10.f, 10000.f,
                  false );
               LOG_INFO(1, "Added material [" << id << "] index=" << m.index << "/" << materialId << " " << 
                  "( " << m.Kd.x << ", " << m.Kd.y << ", " << m.Kd.z << ") " <<
                  "( " << m.Ks.x << ", " << m.Ks.y << ", " << m.Ks.z << ") " <<
                  ", Texture ID= " << m.textureId );
            }
            id = line.substr(7);
            MaterialMTL material;
            memset( &material, 0, sizeof(MaterialMTL));
            material.index = static_cast<unsigned int>(materials.size()+materialId);
            materials[id].textureId = MATERIAL_NONE;
            materials[id] = material;
         }

         if( line.find("Kd") == 0 )
         {
            // RGB Color
            line = line.substr(3);
            materials[id].Kd = readfloat3(line);
         }

         if( line.find("Ks") == 0 )
         {
            // Specular values
            line = line.substr(3);
            materials[id].Ks = readfloat3(line);
         }

         if( line.find("map_Kd") == 0 )
         {
            line = line.substr(7);
            std::string folder(filename);
            size_t backSlashPos = filename.rfind('/');
            if( backSlashPos==-1 )
            {
               backSlashPos = filename.rfind('\\');
            }
            if( backSlashPos != -1 )
            {
               folder = filename.substr(0, backSlashPos);
            }
            folder += '/';
            folder += line;
            int idx = kernel.getNbActiveTextures();
            LOG_INFO(1, "Loading texture " << folder );
            if( kernel.loadTextureFromFile(idx, folder) )
            {
               materials[id].textureId = idx;
               LOG_INFO(1, "Texture successfully loaded into slot " << idx << " and assigned to material " << id << "(" << materials[id].index << ")" );
            }
         }

         if( line.find("d") == 0 || line.find("Tr") == 0 )
         {
            // Specular values
            line = line.substr(2);
            float d=static_cast<float>(atof(line.c_str()));
            if( d!=0.f )
            {
               materials[id].reflection   = 1.f; 
               materials[id].transparency = d; 
               materials[id].refraction   = 1.66f;
            }
         }
      }

      if( id.length() != 0 )
      {
         MaterialMTL& m = materials[id];
         kernel.setMaterial(
            m.index,
            m.Kd.x,m.Kd.y,m.Kd.z,
            0.f,m.reflection,m.refraction, false, false, 0,
            m.transparency, m.textureId,
            m.Ks.x, 200.f*m.Ks.y, m.Ks.z,
            0.f, 10.f, 100000.f,
            false );
         LOG_INFO(1, "Added material [" << id << "] index=" << m.index << "/" << materialId << " " << 
            "( " << m.Kd.x << ", " << m.Kd.y << ", " << m.Kd.z << ") " <<
            "( " << m.Ks.x << ", " << m.Ks.y << ", " << m.Ks.z << ") " <<
            ", Texture ID= " << m.textureId );
      }

      file.close();
   }
   return 0;
}

float3 OBJReader::loadModelFromFile(
   const std::string& filename,
   GPUKernel& kernel,
   const float3& center,
   const bool autoScale,
   const float& scale,
   bool loadMaterials,
   int materialId,
   bool allSpheres)
{
   std::map<int,float3> vertices;
   std::map<int,float3> normals;
   std::map<int,float3> textureCoordinates;
   std::map<std::string,MaterialMTL> materials;

   float3 minPos = { 100000.f, 100000.f, 100000.f };
   float3 maxPos = {-100000.f,-100000.f,-100000.f };

   std::string noExtFilename(filename);
   size_t pos(noExtFilename.find(".obj"));
   if( pos != -1 )
   {
      noExtFilename = filename.substr(0, pos);
   }

   // Load materials
   if( loadMaterials )
   {
      loadMaterialsFromFile( noExtFilename, materials, kernel, materialId );
   }

   // Load model
   std::string modelFilename(noExtFilename);
   modelFilename += ".obj";

   int index_vertices(1);
   int index_normals(1);
   int index_textureCoordinates(1);
   std::ifstream file(modelFilename.c_str());
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         line.erase( std::remove(line.begin(), line.end(), '\r'), line.end());
         if( line.length() > 1 ) 
         {
            if( line[0] == 'v' )
            {
               // Vertices
               float3 vertex = {0.f,0.f,0.f};
               std::string value("");

               size_t i(1);
               int item(0);
               char previousChar = line[0];
               while( i<line.length() && item<4)
               {
                  if( line[i] == ' '  && previousChar != ' ') 
                  {
                     switch( item )
                     {
                     case 1: vertex.x = static_cast<float>(atof(value.c_str())); break;
                     case 2: vertex.y = static_cast<float>(atof(value.c_str())); break;
                     case 3: vertex.z = static_cast<float>(atof(value.c_str())); break;
                     }
                     ++item;
                     value = "";
                  }
                  else
                  {
                     value += line[i];
                  }

                  previousChar = line[i];
                  ++i;
               }

               if( value.length() != 0 )
               {
                  switch( item )
                  {
                  case 2: vertex.y = static_cast<float>(atof(value.c_str())); break;
                  case 3: vertex.z = static_cast<float>(atof(value.c_str())); break;
                  }
               }

               if( line[1] == 'n' )
               {  
                  // Normals
                  vertex.z = -vertex.z;
                  normals[index_normals] = vertex;
                  ++index_normals;
               }
               else if( line[1] == 't' )
               {  
                  // Texture coordinates
                  textureCoordinates[index_textureCoordinates] = vertex;
                  ++index_textureCoordinates;
               }
               else
               {
                  // Vertex
                  if( line[1] == ' ' ) 
                  {
                     vertex.z = -vertex.z;
                     vertices[index_vertices] = vertex;
                     ++index_vertices;

                     // min
                     minPos.x = (vertex.x < minPos.x) ? vertex.x : minPos.x;
                     minPos.y = (vertex.y < minPos.y) ? vertex.y : minPos.y;
                     minPos.z = (vertex.z < minPos.z) ? vertex.z : minPos.z;
             
                     // max
                     maxPos.x = (vertex.x > maxPos.x) ? vertex.x : maxPos.x;
                     maxPos.y = (vertex.y > maxPos.y) ? vertex.y : maxPos.y;
                     maxPos.z = (vertex.z > maxPos.z) ? vertex.z : maxPos.z;
                  }
               }

            }
         }
      }
      file.close();
   }

   LOG_INFO(3, "Nb Vertices: " << vertices.size());
   LOG_INFO(3, "Nb Normals : " << normals.size());
   //float objectScale = (scale/std::max( maxPos.x - minPos.x, std::max ( maxPos.y - minPos.y, maxPos.z - minPos.z )));
   //float objectScale = (scale/std::max ( maxPos.y - minPos.y, maxPos.x - minPos.x ));

   float3 objectCenter = center;
   float objectScale = scale;
   if( autoScale )
   {
      objectScale = scale/(maxPos.y - minPos.y);

      // Center align object
      objectCenter.x = (minPos.x+maxPos.x) / 2.f;
      objectCenter.y = (minPos.y+maxPos.y) / 2.f;
      objectCenter.z = (minPos.z+maxPos.z) / 2.f;
      file.close();
      LOG_INFO(3, "Min   : " << minPos.x << "," << minPos.y << "," << minPos.z );
      LOG_INFO(3, "Max   : " << maxPos.x << "," << maxPos.y << "," << maxPos.z );
      LOG_INFO(3, "Center: " << objectCenter.x << "," << objectCenter.y << "," << objectCenter.z );
      LOG_INFO(3, "Scale : " << objectScale );
   }

   // Populate ray-tracing engine
   file.open(modelFilename.c_str());
   if( file.is_open() )
   {
      int material = materialId;
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         line.erase( std::remove(line.begin(), line.end(), '\r'), line.end());
         if( line.length() != 0 ) 
         {
            if( line.find("usemtl") == 0 && line.length()>7)
            {
               std::string value = line.substr(7);
               //std::cout << "Material [" << value << "]: ";
               if( materials.find(value) != materials.end() )
               {
                  MaterialMTL& m=materials[value];
                  material = m.index;
                  //std::cout << material << std::endl;
               }
               /*
               else
               {
                  std::cout << "<undefined>" << std::endl;
               }
               */
            }

            if( line[0] == 'f' )
            {
               std::string value("");

               std::vector<int4> face;
               size_t i(1);
               int item(0);
               char previousChar = line[0];
               while( i<line.length() )
               {
                  if(line[i] != ' ') value += line[i];
                  if( i==(line.length()-1) || (line[i] == ' ' && previousChar != ' ')) 
                  {
                     if( value.length() != 0 )
                     {
                        int4 v = readFace(value);
                        face.push_back(v);
                        value = "";
                     }
                  }

                  previousChar = line[i];
                  ++i;
               }

               int f(0);
               int nbPrimitives(0);
               if( allSpheres )
               {
                  float3 sphereCenter;
                  sphereCenter.x = (vertices[face[f].x].x + vertices[face[f+1].x].x + vertices[face[f+2].x].x)/3.f;
                  sphereCenter.y = (vertices[face[f].x].y + vertices[face[f+1].x].y + vertices[face[f+2].x].y)/3.f;
                  sphereCenter.z = (vertices[face[f].x].z + vertices[face[f+1].x].z + vertices[face[f+2].x].z)/3.f;

                  float3 sphereRadius;
                  sphereRadius.x = sphereCenter.x - vertices[face[f].x].x;
                  sphereRadius.y = sphereCenter.y - vertices[face[f].x].y;
                  sphereRadius.z = sphereCenter.z - vertices[face[f].x].z;
                  
                  float radius = 100.f; //objectScale*sqrt(sphereRadius.x*sphereRadius.x+sphereRadius.y*sphereRadius.y+sphereRadius.z*sphereRadius.z);

                  nbPrimitives = kernel.addPrimitive( ptSphere );
                  kernel.setPrimitive( 
                     nbPrimitives,
                     center.x+objectScale*(-objectCenter.x+sphereCenter.x),center.y+objectScale*(-objectCenter.y+sphereCenter.y),center.z+objectScale*(-objectCenter.z+sphereCenter.z),
                     radius, 0.f, 0.f,
                     material);
               }
               else
               {
                  nbPrimitives = kernel.addPrimitive( ptTriangle );
                  kernel.setPrimitive( 
                     nbPrimitives,
                     center.x+objectScale*(-objectCenter.x+vertices[face[f  ].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f  ].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f  ].x].z),
                     center.x+objectScale*(-objectCenter.x+vertices[face[f+1].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+1].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+1].x].z),
                     center.x+objectScale*(-objectCenter.x+vertices[face[f+2].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+2].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+2].x].z),
                     0.f, 0.f, 0.f,
                     material);
               }

               // Texture coordinates
               kernel.setPrimitiveTextureCoordinates( nbPrimitives, textureCoordinates[face[f].y], textureCoordinates[face[f+1].y], textureCoordinates[face[f+2].y] );
               
               // Normals
               if( face[f].z!=0 && face[f+1].z!=0 && face[f+2].z!=0 )
               {
                  kernel.setPrimitiveNormals( nbPrimitives, normals[face[f].z], normals[face[f+1].z], normals[face[f+2].z] );
               }

               if( face.size() == 4 )
               {
                  if( allSpheres )
                  {
                     float3 sphereCenter;
                     sphereCenter.x = (vertices[face[f+3].x].x + vertices[face[f+2].x].x + vertices[face[f].x].x)/3.f;
                     sphereCenter.y = (vertices[face[f+3].x].y + vertices[face[f+2].x].y + vertices[face[f].x].y)/3.f;
                     sphereCenter.z = (vertices[face[f+3].x].z + vertices[face[f+2].x].z + vertices[face[f].x].z)/3.f;

                     float3 sphereRadius;
                     sphereRadius.x = sphereCenter.x - vertices[face[f].x].x;
                     sphereRadius.y = sphereCenter.y - vertices[face[f].x].y;
                     sphereRadius.z = sphereCenter.z - vertices[face[f].x].z;
                  
                     float radius = 100.f; //objectScale*sqrt(sphereRadius.x*sphereRadius.x+sphereRadius.y*sphereRadius.y+sphereRadius.z*sphereRadius.z);

                     nbPrimitives = kernel.addPrimitive( ptSphere );
                     kernel.setPrimitive( 
                        nbPrimitives,
                        center.x+objectScale*(-objectCenter.x+sphereCenter.x),center.y+objectScale*(-objectCenter.y+sphereCenter.y),center.z+objectScale*(-objectCenter.z+sphereCenter.z),
                        radius, 0.f, 0.f,
                        material);
                  }
                  else
                  {
                     nbPrimitives = kernel.addPrimitive( ptTriangle );
                     kernel.setPrimitive( 
                        nbPrimitives, 
                        center.x+objectScale*(-objectCenter.x+vertices[face[f+3].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+3].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+3].x].z),
                        center.x+objectScale*(-objectCenter.x+vertices[face[f+2].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+2].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+2].x].z),
                        center.x+objectScale*(-objectCenter.x+vertices[face[f  ].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f  ].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f  ].x].z),
                        0.f, 0.f, 0.f,
                        material);
                  }
                  // Texture coordinates
                  kernel.setPrimitiveTextureCoordinates( nbPrimitives, textureCoordinates[face[f+3].y], textureCoordinates[face[f+2].y], textureCoordinates[face[f].y] );
                  if( face[f].z!=0 && face[f+2].z!=0 && face[f+3].z!=0 )
                  {
                     kernel.setPrimitiveNormals( nbPrimitives, normals[face[f+3].z], normals[face[f+2].z], normals[face[f].z] );
                  }
               }
            }
         }
      }
      file.close();
   }
   float3 objectSize;
   objectSize.x = (maxPos.x - minPos.x)*objectScale;
   objectSize.y = (maxPos.y - minPos.y)*objectScale;
   objectSize.z = (maxPos.z - minPos.z)*objectScale;
   
   LOG_INFO( 1, "Loading " << modelFilename.c_str() << " into frame " << kernel.getFrame() << " [" << kernel.getNbActivePrimitives() << " primitives]" );

   kernel.setNbMaxPrimitivePerBox( 2*static_cast<int>(sqrt(static_cast<float>(kernel.getNbActivePrimitives()))));

   return objectSize;
}
