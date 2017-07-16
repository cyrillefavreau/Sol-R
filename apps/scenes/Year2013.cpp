/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include "Year2013.h"

Year2013::Year2013( const std::string& name, const int nbMaxPrimitivePerBox  )
 : Scene( name, nbMaxPrimitivePerBox )
{
}

Year2013::~Year2013(void)
{
}

void Year2013::doInitialize()
{
   /*
   // EPFL
   int g2013[10][23] = 
   {
      { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
      { 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
      { 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0 },
      { 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0 }
   };

   // GTC
   int g2013[10][23] = 
   {
      { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
      { 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
      { 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
      { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
      { 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0 },
      { 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0 }
   };
   */

   // CERN
   int g2013[10][23] = 
   {
      { 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1 },
      { 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1 },
      { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1 },
      { 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1 },
      { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
      { 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1 },
      { 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1 },
      { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0 },
      { 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0 }
   };

   float size = 250.f;
   float X = -(15.f*size);
   float Y = (5.f*size);
   float S = size*1.5f;

	for( int x(0); x<23; ++x)
   {
   	for( int y(0); y<10; ++y)
	   {
         switch( g2013[y][x] )
         {
            case 1:
            {
   			   m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
			      m_gpuKernel->setPrimitive(
                  m_nbPrimitives, 
                  X+x*S, 
                  Y-y*S,
                  0.f, 
                  size*1.2f, 0.f, 0.f, 
                  1030+(x/6)*5 ); 
               break;
            }
            case 2:
            {
   			   m_nbPrimitives = m_gpuKernel->addPrimitive( ptEllipsoid );
			      m_gpuKernel->setPrimitive(
                  m_nbPrimitives, 
                  X+x*S, 
                  Y-y*S,
                  0.f, 
                  size*4.f, size, size*4.f, 
                  1040 ); 
               break;
            }
            case 3:
            {
   			   m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder );
			      m_gpuKernel->setPrimitive(
                  m_nbPrimitives, 
                  X+x*S, 
                  Y-(y+3)*S,
                  0.f, 
                  X+x*S, 
                  Y-(y-1)*S,
                  0.f, 
                  size*4.f, 0.f, 0.f, 
                  1040 ); 
               break;
            }
         }
      }
	}

	Vertex center = { -2000.f, -1000.f, -8000.f };

   // Ai
	center.x = 3000.f;
	center.y = m_groundHeight+2100.f;
	center.z =-4000.f;
	createWorm(center,1030,rand()%20+1030);

   Vertex angles;
   angles.x = 0.f;
   angles.y = 2.5f;
   angles.z = 0.f;
   m_gpuKernel->rotatePrimitives(center, angles );

   m_gpuKernel->getSceneInfo().nbRayIterations = 20;
}

void Year2013::doAnimate()
{
}

void Year2013::doAddLights()
{
	// lights
   float size = 50.f;
	m_nbPrimitives = m_gpuKernel->addPrimitive(  ptSphere );  
   m_gpuKernel->setPrimitive(  m_nbPrimitives, 5000.f, 5000.f, -5000.f,  size, size*2.f, 0.f, DEFAULT_LIGHT_MATERIAL); 
}
