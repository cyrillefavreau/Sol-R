/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include "DoggyStyleScene.h"

DoggyStyleScene::DoggyStyleScene( const std::string& name, const int nbMaxPrimitivePerBox  )
 : Scene( name, nbMaxPrimitivePerBox )
{
}

DoggyStyleScene::~DoggyStyleScene(void)
{
}


void DoggyStyleScene::doInitialize()
{
	float size = 400.f;

   Vertex center = { -3500.f, m_groundHeight+2500.f, 0.f };
	createDog( center, rand()%20+30, 500.f, 1010 );
	center.x = 3500.f;
	createDog( center, rand()%20+30, 500.f, 1011 );
   Vertex angles = { 0.f, static_cast<float>(M_PI), 0.f };
   m_gpuKernel->rotatePrimitives( center, angles );
}

void DoggyStyleScene::doAnimate()
{
}

void DoggyStyleScene::doAddLights()
{
	// Laser
	for( int i(0); i<3; ++i ) 
	{
      Vertex p1,p2;
      p1.x = rand()%10000-5000.f;
      p1.y = 5000.f;
      p1.z = rand()%10000-5000.f;
      p2.x = -(rand()%10000-5000.f);
      p2.y = -5000.f;
      p2.z = -(rand()%10000-5000.f);
		m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder ); 
		m_gpuKernel->setPrimitive( 
         m_nbPrimitives, 
		 	p1.x, p1.y, p1.z, 
		   p2.x, p2.y, p2.z,
			100.f, 100.f, 100.f, 
         DEFAULT_LIGHT_MATERIAL-i);
	}
}
