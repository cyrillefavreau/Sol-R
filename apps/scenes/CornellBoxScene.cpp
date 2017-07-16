/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include "CornellBoxScene.h"

CornellBoxScene::CornellBoxScene( const std::string& name, const int nbMaxPrimitivePerBox  )
 : Scene( name, nbMaxPrimitivePerBox )
{
}

CornellBoxScene::~CornellBoxScene(void)
{
}

/*
________________________________________________________________________________

Create simple scene with cylinders
________________________________________________________________________________
*/
void CornellBoxScene::doInitialize()
{
   m_groundHeight = -5000.f;
	// Spheres
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); m_gpuKernel->setPrimitive( m_nbPrimitives,  2200.f, 0.f, 0.f, 2000.f, 0.f, 0.f, rand()%50); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); m_gpuKernel->setPrimitive( m_nbPrimitives, -2200.f, 0.f, 0.f, 2000.f, 0.f, 0.f, rand()%50); 

   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); m_gpuKernel->setPrimitive( m_nbPrimitives,  0.f,  2200.f, 0.f, 2000.f, 0.f, 0.f, rand()%50); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); m_gpuKernel->setPrimitive( m_nbPrimitives,  0.f, -2200.f, 0.f, 2000.f, 0.f, 0.f, rand()%50); 

   m_gpuKernel->getSceneInfo().nbRayIterations = 20;
}

void CornellBoxScene::doAnimate()
{
   m_rotationAngles.x = 0.02f;
   m_rotationAngles.y = 0.01f;
   m_rotationAngles.z = 0.015f;
   m_gpuKernel->rotatePrimitives( m_rotationCenter, m_rotationAngles );
   m_gpuKernel->compactBoxes(false);
}

void CornellBoxScene::doAddLights()
{
   for( int i(0); i<5; ++i)
   {
	   // lights
	   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );  
      m_gpuKernel->setPrimitive( m_nbPrimitives,  rand()%20000-10000.f, rand()%5000-m_groundHeight, rand()%20000-10000.f,  10.f, 0.f, 0.f, 120+i); 
      m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );
   }

}
