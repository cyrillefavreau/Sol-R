#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <stdio.h>

#include "Consts.h"
#include "OBJReader.h"

const int OPTIMAL_NB_OF_BOXES = 512;

OBJReader::OBJReader(void)
{
}

OBJReader::~OBJReader(void)
{
}

int OBJReader::processBoxes( std::map<int,float4>& vertices, std::map<int,int>& primitivesPerBox, const int boxSize, int& nbActiveBoxes )
{
   primitivesPerBox.clear();
   float4 boxSteps;
   boxSteps.x = ( m_maxPos.x - m_minPos.x ) / boxSize;
   boxSteps.y = ( m_maxPos.y - m_minPos.y ) / boxSize;
   boxSteps.z = ( m_maxPos.z - m_minPos.z ) / boxSize;
   //std::cout << "Steps " << boxSteps.x << "," << boxSteps.y << "," << boxSteps.z  << std::endl;

   boxSteps.x = ( boxSteps.x == 0.f ) ? 1 : boxSteps.x;
   boxSteps.y = ( boxSteps.y == 0.f ) ? 1 : boxSteps.y;
   boxSteps.z = ( boxSteps.z == 0.f ) ? 1 : boxSteps.z;

   //std::cout << "min " << m_minPos.x << "," << m_minPos.y << "," << m_minPos.z  << std::endl;
   //std::cout << "max " << m_maxPos.x << "," << m_maxPos.y << "," << m_maxPos.z  << std::endl;
   //std::cout << "box [" << boxSize << "] " << boxSteps.x << "," << boxSteps.y << "," << boxSteps.z  << std::endl;

   nbActiveBoxes = 0;
   std::map<int,float4>::iterator it = vertices.begin();
   while( it != vertices.end() )
   {
      float4& vertex((*it).second);

      int X = static_cast<int>(( vertex.x - m_minPos.x ) / boxSteps.x);
      int Y = static_cast<int>(( vertex.y - m_minPos.y ) / boxSteps.y);
      int Z = static_cast<int>(( vertex.z - m_minPos.z ) / boxSteps.z);
      int B = X*boxSize*boxSize + Y*boxSize + Z;

      if( primitivesPerBox.find(B) == primitivesPerBox.end() )
      {
         //std::cout << B << std::endl;
         nbActiveBoxes++;
         primitivesPerBox[B] = 0;
      }
      else
      {
         primitivesPerBox[B]++;
      }
      vertex.w = static_cast<float>(B);
      //std::cout << "V [" << index << "] " << vertex.x << "," << vertex.y << "," << vertex.z << "," << vertex.w << std::endl;
      ++it;
   }

   int maxPrimitivePerBox = 0;
   std::map<int,int>::const_iterator itpb = primitivesPerBox.begin();
   while( itpb != primitivesPerBox.end() )
   {
      //std::cout << "Box " << (*itpb).first << " -> " << (*itpb).second << std::endl;
      maxPrimitivePerBox += (*itpb).second;
      ++itpb;
   }
   maxPrimitivePerBox /= primitivesPerBox.size();

   std::cout << "NbMaxPrimitivePerBox[" << boxSize << "], nbBoxes=" << nbActiveBoxes << ", maxPrimitivePerBox=" << maxPrimitivePerBox << ", Ratio=" << abs(OPTIMAL_NB_OF_BOXES-nbActiveBoxes) << "/" << OPTIMAL_NB_OF_BOXES << std::endl;
   return abs(OPTIMAL_NB_OF_BOXES-nbActiveBoxes);
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
   int boxId,
   float scale,
   int materialId)
{
   /*
   for( int i=boxId; i<NB_MAX_BOXES; ++i )
   {
      gpuKernel.resetBox(i,true);
   }
   */

   std::cout << "Loading " << filename.c_str() << std::endl;

   std::map<int,float4> vertices;
   std::map<int,float4> normals;

   m_minPos.x =  100000.f;
   m_minPos.y =  100000.f;
   m_minPos.z =  100000.f;
   m_maxPos.x = -100000.f;
   m_maxPos.y = -100000.f;
   m_maxPos.z = -100000.f;

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
               while( i<line.length() )
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
                     m_minPos.x = (vertex.x < m_minPos.x) ? vertex.x : m_minPos.x;
                     m_minPos.y = (vertex.y < m_minPos.y) ? vertex.y : m_minPos.y;
                     m_minPos.z = (vertex.z < m_minPos.z) ? vertex.z : m_minPos.z;
             
                     // max
                     m_maxPos.x = (vertex.x > m_maxPos.x) ? vertex.x : m_maxPos.x;
                     m_maxPos.y = (vertex.y > m_maxPos.y) ? vertex.y : m_maxPos.y;
                     m_maxPos.z = (vertex.z > m_maxPos.z) ? vertex.z : m_maxPos.z;
                  }
               }

            }
         }
      }
      file.close();
   }

   std::cout << "Nb Vertices: " << vertices.size() << std::endl;
   std::cout << "Nb Normals : " << normals.size() << std::endl;

   scale = (scale/max( m_maxPos.x - m_minPos.x, max ( m_maxPos.y - m_minPos.y, m_maxPos.z - m_minPos.z )));
   std::cout << "Min  : " << m_minPos.x << "," << m_minPos.y << "," << m_minPos.z << std::endl;
   std::cout << "Max  : " << m_maxPos.x << "," << m_maxPos.y << "," << m_maxPos.z << std::endl;
   std::cout << "Scale: " << scale << std::endl;

   float4 objectCenter = {0.f,0.f,0.f,0.f};
   
   // Altitude
   objectCenter.y = abs(m_minPos.y);

   // Bounding boxes
   std::map<int,int> primitivesPerBox;
   int maxPrimitivePerBox(0);
   int boxSize = 32;
   int bestSize = boxSize;
   int bestRatio = 100000;
   int activeBoxes(NB_MAX_BOXES);
   do 
   {
      int ratio = processBoxes( vertices, primitivesPerBox, boxSize, activeBoxes );
      if( ratio < bestRatio ) 
      {
         bestSize = boxSize;
         bestRatio = ratio;
      }
      boxSize -= 2;
   }
   while( boxSize>=2 );
   std::cout << "Best Ratio: " << bestSize << " = " << bestRatio << std::endl;
   processBoxes( vertices, primitivesPerBox, bestSize, activeBoxes );

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
               gpuKernel.setPrimitive( nbPrimitives, boxId + static_cast<int>(vertices[face[f].x].w),
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
                  gpuKernel.setPrimitive( nbPrimitives, boxId + static_cast<int>(vertices[face[f].x].w),
                     center.x+scale*(objectCenter.x+vertices[face[f+3].x].x),center.y+scale*(objectCenter.y+vertices[face[f+3].x].y),center.z+(objectCenter.z+scale*vertices[face[f+3].x].z),
                     center.x+scale*(objectCenter.x+vertices[face[f+2].x].x),center.y+scale*(objectCenter.y+vertices[face[f+2].x].y),center.z+(objectCenter.z+scale*vertices[face[f+2].x].z),
                     center.x+scale*(objectCenter.x+vertices[face[f  ].x].x),center.y+scale*(objectCenter.y+vertices[face[f  ].x].y),center.z+(objectCenter.z+scale*vertices[face[f  ].x].z),
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
   return m_minPos;
}