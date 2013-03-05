#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "Consts.h"
#include "OBJReader.h"

OBJReader::OBJReader(void)
{
}

OBJReader::~OBJReader(void)
{
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

float4 OBJReader::loadModelFromFile(
   const std::string& filename,
   GPUKernel& gpuKernel,
   const float4& center,
   float scale,
   int materialId)
{
   std::cout << "Loading " << filename.c_str() << std::endl;

   std::map<int,float4> vertices;
   std::map<int,float4> normals;

   float4 minPos = { 100000.f, 100000.f, 100000.f, 0.f };
   float4 maxPos = {-100000.f,-100000.f,-100000.f, 0.f };

   int index_vertices(1);
   int index_normals(1);
   std::ifstream file(filename.c_str());
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         if( line.length() > 1 ) 
         {
            if( line[0] == 'v' )
            {
               float4 vertex = {0.f,0.f,0.f,0.f};
               std::string value("");

               size_t i(1);
               int item(0);
               char previousChar = line[0];
               while( i<line.length() /*&& item<4*/)
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
   //scale = (scale/max( maxPos.x - minPos.x, max ( maxPos.y - minPos.y, maxPos.z - minPos.z )));
   scale = (scale/max( maxPos.y - minPos.y, maxPos.z - minPos.z ));

   // Center align object
   float4 objectCenter = {0.f,0.f,0.f,0.f};
   objectCenter.y = abs(minPos.y);

   
   file.open(filename);
   if( file.is_open() )
   {
      int material = 0;
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         if( line.length() != 0 ) 
         {
            if( line[0] == 'u' )
            {
               std::string value = line.substr(16,line.length());
               material = atoi(value.c_str())%10;
               //std::cout << material << std::endl;
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
               int nbPrimitives = gpuKernel.addPrimitive( ptTriangle );
               gpuKernel.setPrimitive( 
                  nbPrimitives,
                  center.x+scale*(objectCenter.x+vertices[face[f  ].x].x),center.y+scale*(objectCenter.y+vertices[face[f  ].x].y),center.z+scale*(objectCenter.z+vertices[face[f  ].x].z),
                  center.x+scale*(objectCenter.x+vertices[face[f+1].x].x),center.y+scale*(objectCenter.y+vertices[face[f+1].x].y),center.z+scale*(objectCenter.z+vertices[face[f+1].x].z),
                  center.x+scale*(objectCenter.x+vertices[face[f+2].x].x),center.y+scale*(objectCenter.y+vertices[face[f+2].x].y),center.z+scale*(objectCenter.z+vertices[face[f+2].x].z),
                  0.f, 0.f, 0.f,
                  materialId+material, 1, 1);
               if( face[f].z!=0 && face[f+1].z!=0 && face[f+2].z!=0 )
               {
                  gpuKernel.setPrimitiveNormals( nbPrimitives, normals[face[f].z], normals[face[f+1].z], normals[face[f+2].z] );
               }

               if( face.size() == 4 )
               {
                  nbPrimitives = gpuKernel.addPrimitive( ptTriangle );
                  gpuKernel.setPrimitive( 
                     nbPrimitives, 
                     center.x+scale*(objectCenter.x+vertices[face[f+3].x].x),center.y+scale*(objectCenter.y+vertices[face[f+3].x].y),center.z+scale*(objectCenter.z+vertices[face[f+3].x].z),
                     center.x+scale*(objectCenter.x+vertices[face[f+2].x].x),center.y+scale*(objectCenter.y+vertices[face[f+2].x].y),center.z+scale*(objectCenter.z+vertices[face[f+2].x].z),
                     center.x+scale*(objectCenter.x+vertices[face[f  ].x].x),center.y+scale*(objectCenter.y+vertices[face[f  ].x].y),center.z+scale*(objectCenter.z+vertices[face[f  ].x].z),
                     0.f, 0.f, 0.f,
                     materialId+material, 1, 1);
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
   return center;
}