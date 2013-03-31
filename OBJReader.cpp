#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>

#include "Consts.h"
#include "OBJReader.h"

OBJReader::OBJReader(void)
{
}

OBJReader::~OBJReader(void)
{
}

float4 readFloat4( const std::string& value )
{
   float4 returnValue = {0.f,0.f,0.f,0.f};
   int item(0);
   std::string tmp;
   for( int i(0); i<value.length(); ++i)
   {
      if( value[i] != ' ') tmp += value[i];

      if( value[i] == ' ' || i==(value.length()-1) )
      {
         switch( item )
         {
            case 0: returnValue.x = atof(tmp.c_str()); break;
            case 1: returnValue.y = atof(tmp.c_str()); break;
            case 2: returnValue.z = atof(tmp.c_str()); break;
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

      if( face[i] == '/' || i==(face.length()-1) )
      {
         switch( item )
         {
            case 0: returnValue.x = abs(atoi(value.c_str())); break;
            case 1: returnValue.y = abs(atoi(value.c_str())); break;
            case 2: returnValue.z = abs(atoi(value.c_str())); break;
         }
         ++item;
         value = "";
      }
   }
   return returnValue;
}

unsigned int OBJReader::loadMaterialsFromFile(
   const std::string& filename,
   std::map<std::string,MaterialMTL>& m_materials,
   GPUKernel& gpuKernel,
   int materialId)
{
   std::string materialsFilename(filename);
   materialsFilename += ".mtl";

   std::string id;
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
               MaterialMTL& m = m_materials[id];
               m.index = (m_materials.size()-1)%50 + materialId;
               gpuKernel.setMaterial(
                  m.index,
                  m.Kd.x,m.Kd.y,m.Kd.z,
                  0.f,m.reflection,m.refraction, false, false, 0,
                  m.transparency, TEXTURE_NONE,
                  m.Ks.x, 200.f, 0.f, // m.Ks.y,m.Ks.z,
                  0.f, false );
               std::cout << "Added material [" << id << "] index=" << m.index << "/" << materialId << " " << 
                  "( " << m.Kd.x << ", " << m.Kd.y << ", " << m.Kd.z << ") " <<
                  "( " << m.Ks.x << ", " << m.Ks.y << ", " << m.Ks.z << ") " <<
                  std::endl;
            }
            id = line.substr(7);
            MaterialMTL material;
            memset( &material, 0, sizeof(MaterialMTL));
            m_materials[id] = material;
         }

         if( line.find("Kd") == 0 )
         {
            // RGB Color
            line = line.substr(3);
            m_materials[id].Kd = readFloat4(line);
         }

         if( line.find("Ks") == 0 )
         {
            // Specular values
            line = line.substr(3);
            m_materials[id].Ks = readFloat4(line);
         }

         if( line.find("d") == 0 || line.find("Tr") == 0 )
         {
            // Specular values
            line = line.substr(2);
            int d=atoi(line.c_str());
            switch(d)
            {
            case 3: 
            case 5: 
            case 8: 
               m_materials[id].reflection   = rand()%100/100.f; 
               break;
            case 4: 
            case 7: 
               m_materials[id].reflection   = rand()%100/100.f; 
               m_materials[id].transparency = rand()%100/100.f; 
               m_materials[id].refraction   = 1.01f+rand()%100/100.f; 
               break;
            case 6: 
            case 9: 
               m_materials[id].transparency = rand()%100/100.f; 
               m_materials[id].refraction   = 1.01f+rand()%100/100.f; 
               break;
            }
         }
      }

      if( id.length() != 0 )
      {
         MaterialMTL& m = m_materials[id];
         m.index = (m_materials.size()-1)%50 + materialId;
         gpuKernel.setMaterial(
            m.index,
            m.Kd.x,m.Kd.y,m.Kd.z,
            0.f,m.reflection,m.refraction, false, false, 0,
            m.transparency, TEXTURE_NONE,
            m.Ks.x, 200.f, 0.f, // m.Ks.y,m.Ks.z,
            0.f, false );
         std::cout << "Added material [" << id << "] index=" << m.index << "/" << materialId << " " << 
            "( " << m.Kd.x << ", " << m.Kd.y << ", " << m.Kd.z << ") " <<
            "( " << m.Ks.x << ", " << m.Ks.y << ", " << m.Ks.z << ") " <<
            std::endl;
      }

      file.close();
   }
   return 0;
}

float4 OBJReader::loadModelFromFile(
   const std::string& filename,
   GPUKernel& gpuKernel,
   const float4& center,
   const float& scale,
   int materialId)
{
   std::map<int,float4> vertices;
   std::map<int,float4> normals;

   float4 minPos = { 100000.f, 100000.f, 100000.f, 0.f };
   float4 maxPos = {-100000.f,-100000.f,-100000.f, 0.f };

   std::string noExtFilename(filename);
   size_t pos(noExtFilename.find(".obj"));
   if( pos != -1 )
   {
      noExtFilename = filename.substr(0, pos);
   }

   // Load materials
   std::map<std::string,MaterialMTL> materials;
   loadMaterialsFromFile( noExtFilename, materials, gpuKernel, materialId );

   // Load model
   std::string modelFilename(noExtFilename);
   modelFilename += ".obj";

   std::cout << "Loading " << modelFilename.c_str() << std::endl;

   int index_vertices(1);
   int index_normals(1);
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
               float4 vertex = {0.f,0.f,0.f,0.f};
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
                     case 3: vertex.z = -static_cast<float>(atof(value.c_str())); break;
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

               if( item != 4 )
               {
                  vertex.z = -static_cast<float>(atof(value.c_str()));
               }

               if( line[1] == 'n' )
               {  
                  normals[index_normals] = vertex;
                  ++index_normals;
               }
               else
               {
                  if( line[1] == ' ' ) 
                  {
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

   std::cout << "Nb Vertices: " << vertices.size() << std::endl;
   std::cout << "Nb Normals : " << normals.size() << std::endl;
   float objectScale = (scale/std::max( maxPos.x - minPos.x, std::max ( maxPos.y - minPos.y, maxPos.z - minPos.z )));

   // Center align object
   float4 objectCenter = {0.f,0.f,0.f,0.f};
   objectCenter.x = (minPos.x+maxPos.x) / 2.f;
   objectCenter.y = (minPos.y+maxPos.y) / 2.f;
   objectCenter.z = (minPos.z+maxPos.z) / 2.f;
   file.close();
   std::cout << "Min   : " << minPos.x << "," << minPos.y << "," << minPos.z << std::endl;
   std::cout << "Max   : " << maxPos.x << "," << maxPos.y << "," << maxPos.z << std::endl;
   std::cout << "Center: " << objectCenter.x << "," << objectCenter.y << "," << objectCenter.z << std::endl;
   std::cout << "Scale : " << objectScale << std::endl;

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
               std::cout << "Material [" << value << "]: ";
               if( materials.find(value) != materials.end() )
               {
                  MaterialMTL& m=materials[value];
                  material = m.index;
                  std::cout << material << std::endl;
               }
               else
               {
                  std::cout << "<undefined>" << std::endl;
               }
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
               //std::cout << "F(" << face[f] << "," << face[f+1] << "," << face[f+2] << ")" << std::endl;
               unsigned int nbPrimitives = gpuKernel.addPrimitive( ptTriangle );
               gpuKernel.setPrimitive( 
                  nbPrimitives,
                  center.x+objectScale*(-objectCenter.x+vertices[face[f  ].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f  ].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f  ].x].z),
                  center.x+objectScale*(-objectCenter.x+vertices[face[f+1].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+1].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+1].x].z),
                  center.x+objectScale*(-objectCenter.x+vertices[face[f+2].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+2].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+2].x].z),
                  0.f, 0.f, 0.f,
                  material, 1, 1);

               if( face[f].z!=0 && face[f+1].z!=0 && face[f+2].z!=0 )
               {
                  gpuKernel.setPrimitiveNormals( nbPrimitives, normals[face[f].z], normals[face[f+1].z], normals[face[f+2].z] );
               }

               if( face.size() == 4 )
               {
                  nbPrimitives = gpuKernel.addPrimitive( ptTriangle );
                  gpuKernel.setPrimitive( 
                     nbPrimitives, 
                     center.x+objectScale*(-objectCenter.x+vertices[face[f+3].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+3].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+3].x].z),
                     center.x+objectScale*(-objectCenter.x+vertices[face[f+2].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f+2].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f+2].x].z),
                     center.x+objectScale*(-objectCenter.x+vertices[face[f  ].x].x),center.y+objectScale*(-objectCenter.y+vertices[face[f  ].x].y),center.z+objectScale*(-objectCenter.z+vertices[face[f  ].x].z),
                     0.f, 0.f, 0.f,
                     material, 1, 1);
                  if( face[f].z!=0 && face[f+2].z!=0 && face[f+3].z!=0 )
                  {
                     gpuKernel.setPrimitiveNormals( nbPrimitives, normals[face[f+3].z], normals[face[f+2].z], normals[face[f].z] );
                  }
               }
            }
         }
      }
      file.close();
   }
   float4 objectSize;
   objectSize.x = (maxPos.x - minPos.x)*objectScale;
   objectSize.y = (maxPos.y - minPos.y)*objectScale;
   objectSize.z = (maxPos.z - minPos.z)*objectScale;
   return objectSize;
}
