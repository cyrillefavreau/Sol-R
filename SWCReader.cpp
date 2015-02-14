/* 
* Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
*/

#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string.h>
#include <math.h>

#include "Logging.h"

#include "Consts.h"
#include "SWCReader.h"

struct Morphology
{
   int branch;
   float x;
   float y;
   float z;
   float radius;
   int parent;
};

typedef std::map<int, Morphology> Morphologies;

SWCReader::SWCReader(void)
{
}

SWCReader::~SWCReader(void)
{
}

CPUBoundingBox SWCReader::loadMorphologyFromFile(
   const std::string& filename,
   GPUKernel& kernel,
   const Vertex& center,
   const bool autoScale,
   const Vertex& scale,
   bool autoCenter,
   const int materialId )
{
   Morphologies morphologies;

   CPUBoundingBox AABB;
   LOG_INFO(1,"Loading SWC file " << filename );

   Vertex ObjectSize = {0.f,0.f,0.f};

   // Read vertices
   std::ifstream file(filename.c_str());
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );

         std::string A,B,C,D,E,F,G;
         file >> A >> B >> C >> D >> E >> F >> G;
         if( A!="#" )
         {
            Morphology morphology;
            int id = atoi(A.c_str());
            morphology.branch = atoi(B.c_str());
            morphology.x = static_cast<float>(scale.x*(center.x+atof(C.c_str())));
            morphology.y = static_cast<float>(scale.y*(center.y+atof(D.c_str())));
            morphology.z = static_cast<float>(scale.z*(center.z+atof(E.c_str())));
            morphology.radius = static_cast<float>(scale.w*atof(F.c_str()));
            morphology.parent = atoi(G.c_str());
            morphologies[id] = morphology;
         }
      }
      file.close();
   }

   Morphologies::iterator it = morphologies.begin();
   while( it!=morphologies.end() )
   {
      Morphology& a = (*it).second;
      if( a.parent == -1 )
      {
         Vertex vt0 = {0.f,0.f,0.f};
         Vertex vt1 = {2.f,2.f,0.f};
         Vertex vt2 = {0.f,0.f,0.f};

         int p = kernel.addPrimitive(ptSphere,true);
         kernel.setPrimitive(p, a.x, a.y, a.z, a.radius*1.2f, 0.f, 0.f, materialId );
         kernel.setPrimitiveTextureCoordinates(p, vt0, vt1, vt2);
      }
      else
      {
         Morphology& b = morphologies[a.parent];
         if( b.parent != -1 )
         {
            Vertex vt0 = {0.f,0.f,0.f};
            Vertex vt1 = {1.f,1.f,0.f};
            Vertex vt2 = {0.f,0.f,0.f};

            float ra = a.radius*2.f;
            float rb = b.radius*2.f;
            int p = kernel.addPrimitive(ptCylinder,true);
            kernel.setPrimitive(p, a.x, a.y, a.z, b.x, b.y, b.z, ra, 0.f, 0.f, materialId );
            kernel.setPrimitiveTextureCoordinates(p, vt0, vt1, vt2);
            p = kernel.addPrimitive(ptSphere,true);
            kernel.setPrimitive(p, b.x, b.y, b.z, rb, 0.f, 0.f, materialId );
            kernel.setPrimitiveTextureCoordinates(p, vt0, vt1, vt2);
         }
      }
      ++it;
   }

   LOG_INFO(1,"--------------------------------------------------------------------------------");
   LOG_INFO(1, "Loaded " << filename.c_str() << " into frame " << kernel.getFrame() << " [" << kernel.getNbActivePrimitives() << " primitives]" );
   LOG_INFO(1, "inAABB     : (" << AABB.parameters[0].x << "," << AABB.parameters[0].y << "," << AABB.parameters[0].z << "),(" << AABB.parameters[1].x << "," << AABB.parameters[1].y << "," << AABB.parameters[1].z << ")" );

   LOG_INFO(1,"--------------------------------------------------------------------------------");
   return AABB;
}
