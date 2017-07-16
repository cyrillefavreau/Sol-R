/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#include <math.h>
#include <stdlib.h>

#include "XmasScene.h"

XmasScene::XmasScene( const std::string& name, const int nbMaxPrimitivePerBox  )
 : Scene( name, nbMaxPrimitivePerBox )
{
}

XmasScene::~XmasScene(void)
{
}

/*
________________________________________________________________________________

Create tree
________________________________________________________________________________
*/
void XmasScene::createTree( int iteration, int boxId, int maxIterations, Vertex center, int material, float interval, float radius )
{
	if( iteration > 0 )
	{
		for( int i(0); i<(2+rand()%3); ++i )
		{
			Vertex a = center;
			Vertex b = a;
			b.y += interval;

			int box = (iteration<maxIterations) ? boxId : i;

			m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder );
			m_gpuKernel->setPrimitive( 
				m_nbPrimitives,
				a.x, a.y, a.z,
				b.x, b.y, b.z,
				radius/2.f, 0.f, 0.f,
				material);

#if 1
			Vertex angles = { 
				((i==0) ? 0.5f : 2.f)*(rand()%100/100.f-0.5f), 
				((i==0) ? 0.5f : 2.f)*(rand()%100/100.f-0.5f), 
				((i==0) ? 0.5f : 2.f)*(rand()%100/100.f-0.5f) };
				Vertex cosAngles = { cos( angles.x ), cos( angles.y ), cos( angles.z ) };
				Vertex sinAngles = { sin( angles.x ), sin( angles.y ), sin( angles.z ) };
            CPUPrimitive* p = m_gpuKernel->getPrimitive( m_nbPrimitives );
				m_gpuKernel->rotatePrimitive( *p, a, cosAngles, sinAngles );
#endif // 0
				//b.x += interval;
				b.y += interval;
				//b.z += interval;
				m_gpuKernel->getPrimitiveOtherCenter( m_nbPrimitives, b );

				m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
				m_gpuKernel->setPrimitive( 
					m_nbPrimitives, 
					b.x, b.y, b.z,
					radius/2.f, 0.f, 0.f,
					material);

				if( iteration == 1 && rand()%3==0 ) 
				{
					// Boule de noel
					m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder );
					m_gpuKernel->setPrimitive( 
						m_nbPrimitives,
						a.x, a.y, a.z,
						a.x, a.y-300.f, a.z,
						5, 0.f, 0.f,
                  1016);
					m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
					m_gpuKernel->setPrimitive( 
						m_nbPrimitives, 
						a.x, a.y-400.f, a.z,
						200, 0.f, 0.f,
						1010+rand()%6);
				}

#if 1
				if( iteration == 1 && rand()%3==0 ) 
				{
					// Leaves
					m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
					m_gpuKernel->setPrimitive( 
						m_nbPrimitives, 
						b.x, b.y, b.z,
						50.f+rand()%500, 0.f, 0.f,
						1002);
				}
#endif // 0
				createTree( iteration-1, box, maxIterations, b, material, 
					interval*(0.8f+rand()%iteration/10.f), 
					radius*(0.4f+rand()%iteration/10.f) );
		}
	}
}

void XmasScene::doInitialize()
{
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -4200.f, 0.f, 0.f,  0.f, 2000.f, 200.f,    1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXYPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -4100.f, 0.f, 200.f,  100.f, 2000.f,  0.f, 1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXYPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -4100.f, 0.f,-200.f,  100.f, 2000.f,  0.f, 1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, -4100.f, 2000.f,0.f,  100.f, 0.f, 200.f,   1000); 

   m_nbPrimitives = m_gpuKernel->addPrimitive( ptYZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,  4200.f, 0.f, 0.f,  0.f, 2000.f, 200.f,    1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXYPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,  4100.f, 0.f, 200.f,  100.f, 2000.f,  0.f, 1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXYPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,  4100.f, 0.f,-200.f,  100.f, 2000.f,  0.f, 1000); 
   m_nbPrimitives = m_gpuKernel->addPrimitive( ptXZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives,  4100.f, 2000.f,0.f,  100.f, 0.f, 200.f,   1000); 

	int material = 1001;

	// Tree
	m_nbPrimitives = m_gpuKernel->addPrimitive( ptCylinder  );  m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, m_groundHeight, 0.f, 0.f, -1000.f, 0.f, 250.f,  0.f, 0.f, material); 
	m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere    );  m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, -1000.f, 0.f, 250.f,  0.f, 0.f, material); 
	int nbIterations = 2+rand()%3;
	Vertex center = { 0.f, -1000.f, 0.f };
	createTree( nbIterations, 10, nbIterations, center, 
		material, 
		1000.f,//+rand()%800, 
		200.f//+rand()%300 
		);
}

void XmasScene::doAnimate()
{
}

void XmasScene::doAddLights()
{
	// lights
   float size(50.f);
	//m_nbPrimitives = m_gpuKernel->addPrimitive( ptXZPlane );  m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, 5000.f, 0.f,  size, 0.f, size*2.f, DEFAULT_LIGHT_MATERIAL); 
   m_nbPrimitives = m_gpuKernel->addPrimitive(  ptSphere );  m_gpuKernel->setPrimitive(  m_nbPrimitives, -4000.f, 0.f, 0.f,  0.f, 2000.f, 200.f, DEFAULT_LIGHT_MATERIAL); 
   //m_nbPrimitives = m_gpuKernel->addPrimitive(  ptSphere );  m_gpuKernel->setPrimitive(  m_nbPrimitives,  4000.f, 0.f, 0.f,  0.f, 2000.f, 200.f, LIGHT_MATERIAL_001-rand()%5); 
}
