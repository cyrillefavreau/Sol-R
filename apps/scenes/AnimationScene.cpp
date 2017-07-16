/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <string>

#include <Consts.h>
#include <Logging.h>

#include <io/OBJReader.h>
#include <io/PDBReader.h>
#include <io/FileMarshaller.h>

#include <iostream>

#include "AnimationScene.h"

#undef HAND
#ifdef HAND
const int nbFrames(44);
#else
const int nbFrames(136);
#endif // HAND

const int nbModels(1);
const std::string gModels[nbModels] =
{
   "hand"
};

AnimationScene::AnimationScene( const std::string& name, const int nbMaxPrimitivePerBox  )
   : m_wait(0),Scene( name, nbMaxPrimitivePerBox )
{
   m_currentFrame=0;
   m_forward = true;
}

AnimationScene::~AnimationScene(void)
{
}

void AnimationScene::doInitialize()
{
   int m=30;
   OBJReader objReader;
#if 1
   Vertex objectSize = {3000.f,3000.f,3000.f};
   m_groundHeight = -1500.f;
   Vertex center = { 0.f, -m_groundHeight/objectSize.y, 0.f };

   for( int frame=0; frame<nbFrames; ++frame)
   {
      std::string fileName(m_fileName);
      char tmp[50];
#ifdef HAND
      sprintf_s(tmp,50,"%02d", frame);
      fileName += "./animations/hand/hand_";
#else
      //sprintf_s(tmp,50,"%02d", frame);
#ifdef WIN32
      sprintf_s(tmp,50,"%03d", frame+1);
#else
      sprintf(tmp,"%03d", frame+1);
#endif
      //fileName += "./animations/wooddoll/wooddoll_";
      //fileName += "./animations/ben/ben_";
      //fileName += "./animations/11-09/11-09_000";
      fileName += "./animations/08-10/08-10_000";
      //fileName += "./animations/05-09/05-09_000";
      //fileName += "./animations/06-rabbit/06-rabbit_000";
      //fileName += "./animations/12-01/12-01_000";
#endif // HAND
      fileName += tmp;
      fileName += ".obj";

      m_gpuKernel->setFrame(frame);
      m_gpuKernel->resetFrame();
      CPUBoundingBox aabb;
      CPUBoundingBox inAABB;
      Vertex realSize = objReader.loadModelFromFile( fileName, *m_gpuKernel, center, false, objectSize, (frame==0), m, false, true,aabb,false,inAABB);

      // lights
	   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
      m_gpuKernel->setPrimitive( m_nbPrimitives, -10000.f, 10000.f, -10000.f, 100.f, 100.f, 100.f, DEFAULT_LIGHT_MATERIAL); 
      m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );

      addCornellBox(m_cornellBoxType);
      if(frame!=0) m_gpuKernel->compactBoxes(true);
   }
#else
   Vertex scale  = { 10000.f, 10000.f, 10000.f};
   Vertex center = { 0.f, 0.f, 0.f };
   m_groundHeight = -5000.f;

   m_gpuKernel->setNbFrames(nbFrames);
   //m_gpuKernel->setTreeDepth(8);
   m_gpuKernel->getSceneInfo().nbRayIterations.x = 2;
   //m_gpuKernel->setOptimalNbOfBoxes(16384);

   bool allSpheres=(rand()%2==0);

   for( int f(0); f<nbFrames; ++f )
   {
      m_gpuKernel->setFrame(f);
      m_gpuKernel->resetFrame();
      
      addCornellBox(m_cornellBoxType);

      switch(f)
      {
      case 0:
         {
            std::string filename("./obj/animations/");
            filename += gModels[rand()%nbModels];
            filename += ".obj";
            LOG_INFO(1,"Loading " << filename);

            // lights
	         m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );  
            m_gpuKernel->setPrimitive( m_nbPrimitives, -10000.f, 10000.f, -10000.f, 100.f, 100.f, 100.f, DEFAULT_LIGHT_MATERIAL); 
            m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );

            Vertex realSize = objReader.loadModelFromFile(filename,*m_gpuKernel,center,true,scale,false,m,allSpheres );
            //m_gpuKernel->compactBoxes(true);
            break;
         }
      case (nbFrames-1):
         {
            std::string filename("./obj/animations/");
            filename += gModels[rand()%nbModels];
            filename += ".obj";
            LOG_INFO(1,"Loading " << filename);

            // lights
	         m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );  
            m_gpuKernel->setPrimitive( m_nbPrimitives, 10000.f, 10000.f, -10000.f, 100.f, 100.f, 100.f, DEFAULT_LIGHT_MATERIAL); 
            m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );

            Vertex realSize = objReader.loadModelFromFile( filename, *m_gpuKernel, center, true, scale, false, m, allSpheres );
            m_gpuKernel->compactBoxes(true);
            break;
         }
      }
   }
   m_gpuKernel->morphPrimitives();
#endif // 0
   m_currentFrame = 0;
   m_wait = 0;
   m_gpuKernel->setFrame(m_currentFrame);
}

void AnimationScene::doAnimate()
{
   m_gpuKernel->setFrame(m_currentFrame);
   m_gpuKernel->compactBoxes(false);
#if 0
   if( m_wait>10 )
   {
      m_currentFrame += (m_forward) ? 1 : -1;
      if( m_currentFrame>(nbFrames-2) || m_currentFrame<1)
      {
         m_forward = !m_forward;
         m_wait = 0;
      }
   }
   m_wait++;
#else
   ++m_currentFrame;
   m_currentFrame=m_currentFrame%nbFrames;
#endif
}

void AnimationScene::doAddLights()
{
}
