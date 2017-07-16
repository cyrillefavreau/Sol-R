/* 
* Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
*/

#include <math.h>
#include <stdlib.h>

#ifndef WIN32
#include <dirent.h>
#include <errno.h>
#endif // WIN32

#include <Logging.h>
#include <io/FileMarshaller.h>
#include "SwcScene.h"

SwcScene::SwcScene( const std::string& name, const int nbMaxPrimitivePerBox  )
   : Scene( name, nbMaxPrimitivePerBox )
   , m_counter(0)
{
}

SwcScene::~SwcScene(void)
{
}

/*
________________________________________________________________________________

Create simple scene for box validation
________________________________________________________________________________
*/
void SwcScene::doInitialize()
{
   // initialization
   std::vector<std::string> proteinNames;
   std::string folder("/Users/favreau/medias/morphologies/julian/2cells");
#ifdef WIN32
   // Proteins vector
   HANDLE hFind(nullptr);
   WIN32_FIND_DATA FindData;

   std::string fullFilter;
   fullFilter = folder;
   fullFilter += "*.swc";
   hFind = FindFirstFile(fullFilter.c_str(), &FindData);
   if( hFind != INVALID_HANDLE_VALUE )
   {
      do
      {
         if( strlen(FindData.cFileName) != 0 )
         {
            std::string shortName(FindData.cFileName);
            shortName = shortName.substr(0,shortName.rfind(".swc"));
            proteinNames.push_back(shortName);
         }
      }
      while( FindNextFile(hFind, &FindData) /*&& proteinNames.size()<1*/ );
   }
#else
   DIR *dp;
   struct dirent *dirp;
   if((dp  = opendir(folder.c_str())) == NULL)
   {
      LOG_ERROR(errno << " opening " << folder);
   }
   else
   {
      while ((dirp = readdir(dp)) != NULL)
      {
         std::string filename(dirp->d_name);
         if( filename != "." && filename != ".." &&
            filename.find(".swc") != std::string::npos )
         {
            filename = filename.substr(0,filename.find(".swc"));
            std::string fullPath(folder);
            fullPath += "/";
            fullPath += filename;
            LOG_INFO(1, fullPath);
            proteinNames.push_back(fullPath);
         }
      }
      closedir(dp);
   }
#endif // WIN32

   if( proteinNames.size() != 0 )
   {
      m_currentModel=m_currentModel%proteinNames.size();
      Vertex scale = {100.f,100.f,100.f,1000.f};
      std::string fileName;

      // Scene
      Vertex center = {0.f,0.f,0.f};
      for ( int i(0); i<proteinNames.size(); ++i )
      {
         m_name = proteinNames[i];
         SWCReader swcReader;
#ifdef WIN32
         fileName = folder;
         fileName += m_name;
#else
         fileName = m_name;
#endif // WIN32
         fileName += ".swc";
         int materialId = 1001; //(i%10==0) ? 1001 : 1098;
         LOG_INFO(1, fileName);
         CPUBoundingBox AABB = swcReader.loadMorphologyFromFile(
            fileName, *m_gpuKernel, center, true, scale, true, materialId );

         if( i==0 ) m_morphologies = swcReader.getMorphologies();
         /*
         center.x = rand()%600-300;
         center.y = rand()%600-300;
         center.z = rand()%600-300;
         */
      }
   }

   FileMarshaller fm;
   fm.saveToFile(*m_gpuKernel, "NeuronMorphologies.irt");
}

void SwcScene::doAnimate()
{
   int index = m_counter%m_morphologies.size();
   Morphology& m = m_morphologies[index];
   if( m_counter>0 )
   {
      m_gpuKernel->setPrimitiveMaterial(m_previousPrimitiveId, m_previousMaterial);
   }
   m_previousMaterial = m_gpuKernel->getPrimitiveMaterial(m.primitiveId);
   m_gpuKernel->setPrimitiveMaterial(m.primitiveId, DEFAULT_LIGHT_MATERIAL);
   m_previousPrimitiveId = m.primitiveId;

   int light = m_gpuKernel->getLight(0);
	if (light != -1) {
		CPUPrimitive* lamp = m_gpuKernel->getPrimitive(light);
      lamp->p0.x = m.x;
      lamp->p0.y = m.y;
      lamp->p0.z = m.z - 50.f;
      m_gpuKernel->setPrimitiveCenter(light, lamp->p0);
   }

   m_gpuKernel->compactBoxes(false);
   m_counter += 1;
}

void SwcScene::doAddLights()
{
   // lights
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);  
   m_gpuKernel->setPrimitive( m_nbPrimitives, -10000.f, 10000.f, -10000.f, 20.f, 0.f, 0, DEFAULT_LIGHT_MATERIAL); 
   m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives,false);
}
