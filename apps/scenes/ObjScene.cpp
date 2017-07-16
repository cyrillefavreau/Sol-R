/* 
* Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
*/

//#define USE_NEURONS

#ifdef WIN32
#include <windows.h>
#else
#include <math.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#endif // WIN32

#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <string>

#include <Logging.h>
#include <Consts.h>

#include <io/OBJReader.h>
#include <io/PDBReader.h>
#include <io/FileMarshaller.h>

#include <iostream>

#include "ObjScene.h"

int m_counter = 0;

ObjScene::ObjScene( const std::string& name, const int nbMaxPrimitivePerBox, const std::string& filename )
   : m_filename(filename), Scene( name, nbMaxPrimitivePerBox )
{
   m_currentModel = 0;
   m_groundHeight = -2500.f;
}

ObjScene::~ObjScene(void)
{
}

void ObjScene::doInitialize()
{
#ifdef USE_NEURONS
   const int percentage = 20;
   const float s=10.f;
   const float boxScale = 5.f;
#else
   const float s=10000.f;
#endif
   Vertex objectPosition = {0.f,0.f,0.f};
   m_objectScale.x = s;
   m_objectScale.y = s;
   m_objectScale.z = s;

   // Scene Bounding Box
   SceneInfo& sceneInfo=m_gpuKernel->getSceneInfo();
   CPUBoundingBox AABB;
   AABB.parameters[0].x = sceneInfo.viewDistance;
   AABB.parameters[0].y = sceneInfo.viewDistance;
   AABB.parameters[0].z = sceneInfo.viewDistance;
   AABB.parameters[1].x = -sceneInfo.viewDistance;
   AABB.parameters[1].y = -sceneInfo.viewDistance;
   AABB.parameters[1].z = -sceneInfo.viewDistance;

   if( m_filename.length()!=0)
   {
      LOG_INFO(1,"Loading " << m_filename);
      OBJReader objectReader;
      CPUBoundingBox aabb;
      CPUBoundingBox inAABB;
      memset(&inAABB,0,sizeof(CPUBoundingBox));
      Vertex size = objectReader.loadModelFromFile( m_filename, *m_gpuKernel, objectPosition, true, m_objectScale, true, 1000, false , true, aabb, false, inAABB);
      m_groundHeight = -size.y/2.f*m_objectScale.y-EPSILON;
   }
   else
   {
      std::vector<std::string> fileNames;
#ifdef WIN32
      // filename vector
      HANDLE hFind(nullptr);
      WIN32_FIND_DATA FindData;

      std::string fullFilter("./obj/*.obj");

      hFind = FindFirstFile(fullFilter.c_str(), &FindData);
      if( hFind != INVALID_HANDLE_VALUE )
      {
         do 
         {
            if( strlen(FindData.cFileName) != 0 )
               fileNames.push_back(FindData.cFileName);
         }
         while (FindNextFile(hFind, &FindData));
      }
#else
      std::string path="./obj";
      DIR *dp;
      struct dirent *dirp;
      if((dp  = opendir(path.c_str())) == NULL)
      {
         LOG_ERROR(errno << " opening " << path);
      }
      else
      {
         while ((dirp = readdir(dp)) != NULL)
         {
 			std::string filename(dirp->d_name);
 			if( filename != "." && filename != ".." &&
 				filename.find(".obj") != std::string::npos && filename.find(".mtl") == std::string::npos )
            {
               std::string fullPath(path);
               fullPath += "/";
               fullPath += dirp->d_name;
               LOG_INFO(1,"Model: " << fullPath);
               fileNames.push_back(fullPath);
            }
         }
         closedir(dp);
      }
#endif // WIN32

#ifdef USE_NEURONS
      const std::string star("neuron_66776");
      OBJReader objectReader;
      CPUBoundingBox inAABB;
      memset(&inAABB,0,sizeof(CPUBoundingBox));
      Vertex size = objectReader.loadModelFromFile( "./obj/" + star, *m_gpuKernel, objectPosition, false, m_objectScale, false, 10, false, false, AABB, false, inAABB);

      Vertex halfSize = {
         ( AABB.parameters[1].x - AABB.parameters[0].x ) / 2.f,
         ( AABB.parameters[1].y - AABB.parameters[0].y ) / 2.f,
         ( AABB.parameters[1].z - AABB.parameters[0].z ) / 2.f,
      };

      Vertex boxCenter =
      {
         (AABB.parameters[0].x + AABB.parameters[1].x ) / 2.f,
         (AABB.parameters[0].y + AABB.parameters[1].y ) / 2.f,
         (AABB.parameters[0].z + AABB.parameters[1].z ) / 2.f
      };

      inAABB.parameters[0].x = (boxCenter.x - halfSize.x*boxScale);
      inAABB.parameters[0].y = (boxCenter.y - halfSize.y*boxScale);
      inAABB.parameters[0].z = (boxCenter.z - halfSize.y*boxScale);
      inAABB.parameters[1].x = (boxCenter.x + halfSize.x*boxScale);
      inAABB.parameters[1].y = (boxCenter.y + halfSize.y*boxScale);
      inAABB.parameters[1].z = (boxCenter.z + halfSize.z*boxScale);

      m_actorPosition = boxCenter;
      LOG_INFO(1, "Actor Position: " << m_actorPosition.x << "," << m_actorPosition.y << "," << m_actorPosition.z << ")" );

      for( int i(0); i<fileNames.size(); ++i ) 
      {
         if( rand()%100<percentage )
         {
            m_currentModel=i;
#ifdef WIN32
            m_name = "./obj/";
            m_name += fileNames[m_currentModel];
#else
            m_name = fileNames[m_currentModel];
#endif // WIN32
            LOG_INFO(1,"--- Loading " << m_name << " ---");
            if(m_name.find(star)==-1) 
            {
               CPUBoundingBox aabb;
               objectReader.loadModelFromFile( 
                  m_name, *m_gpuKernel, 
                  objectPosition, 
                  false, m_objectScale, false, 
                  11, false, false, 
                  aabb, true, inAABB);

               m_groundHeight = -size.y/2.f-EPSILON;

               if( aabb.parameters[0].x < AABB.parameters[0].x ) AABB.parameters[0].x = aabb.parameters[0].x;
               if( aabb.parameters[0].y < AABB.parameters[0].y ) AABB.parameters[0].y = aabb.parameters[0].y;
               if( aabb.parameters[0].z < AABB.parameters[0].z ) AABB.parameters[0].z = aabb.parameters[0].z;

               if( aabb.parameters[1].x > AABB.parameters[1].x ) AABB.parameters[1].x = aabb.parameters[1].x;
               if( aabb.parameters[1].y > AABB.parameters[1].y ) AABB.parameters[1].y = aabb.parameters[1].y;
               if( aabb.parameters[1].z > AABB.parameters[1].z ) AABB.parameters[1].z = aabb.parameters[1].z;
            }
         }
      }
   }
#else
            m_currentModel=m_counter%fileNames.size();
#ifdef WIN32
            m_name = "./obj/";
            m_name += fileNames[m_currentModel];
#else
            m_name = fileNames[m_currentModel];
#endif // WIN32
            CPUBoundingBox aabb;
            CPUBoundingBox inAABB;
            memset(&inAABB,0,sizeof(CPUBoundingBox));
            OBJReader objectReader;
            Vertex size = objectReader.loadModelFromFile(
               m_name, *m_gpuKernel, 
               objectPosition, 
               true, m_objectScale,
               true, 1000,
               false, true,
               aabb, false, inAABB);
            m_groundHeight = -size.y/2.f-EPSILON;
            LOG_INFO(1,"Ground height = " << m_groundHeight );
         }
#endif // USE_NEURONS
}

void ObjScene::doPostInitialize()
{
#ifdef USE_NEURONS
   Vertex aabbTranslation;
   aabbTranslation.x = -m_actorPosition.x*m_objectScale.x;
   aabbTranslation.y = -m_actorPosition.y*m_objectScale.y;
   aabbTranslation.z = -m_actorPosition.z*m_objectScale.z;
   LOG_INFO(1, "Translation: " << aabbTranslation.x << "," << aabbTranslation.y << "," << aabbTranslation.z << ")" );
   m_gpuKernel->translatePrimitives(aabbTranslation);
   m_gpuKernel->compactBoxes(false);

   // Save into binary format
   FileMarshaller fm;
   fm.saveToFile(*m_gpuKernel,"neurons.irt");
#endif 
}

void ObjScene::doAnimate()
{
}

void ObjScene::doAddLights()
{
   // lights
   if( m_gpuKernel->getNbActiveLamps()==0 )
   {
      m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); 
      //m_gpuKernel->setPrimitive( m_nbPrimitives, m_objectScale.x*0.7f, m_objectScale.y*1.5f, -m_objectScale.z*0.8f, 1.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL); m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );
      //m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL); m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );
      m_gpuKernel->setPrimitive( m_nbPrimitives, 1000.f, 8000.f, -1000.f, 10.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL); 
   }
}
