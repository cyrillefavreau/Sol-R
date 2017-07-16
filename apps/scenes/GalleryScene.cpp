/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#ifdef WIN32
#include <windows.h>
#else
#include <math.h>
#include <stdlib.h>
#endif // WIN32

#include <Consts.h>

#include <io/FileMarshaller.h>
#include <iostream>

#include "GalleryScene.h"
#include "Logging.h"

GalleryScene::GalleryScene( const std::string& name, const int nbMaxPrimitivePerBox )
    : Scene( name, nbMaxPrimitivePerBox )
{
    m_groundHeight = -2500.f;
}

GalleryScene::~GalleryScene(void)
{
}

void GalleryScene::doInitialize()
{
    // Load photos
    std::string folderName = "./photos/";
    loadTextures(folderName,"*.jpg");

    // Materials
    createRandomMaterials(true,false);


#if 0
    // Load model
    HANDLE hFind(nullptr);
    WIN32_FIND_DATA FindData;

    std::vector<std::string> fileNames;
    std::string fullFilter("./irt/*.irt");
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
    Vertex center = {0.f,0.f,-2000.f};
    if( fileNames.size() != 0 )
    {
        m_currentModel=m_currentModel%fileNames.size();
        m_name = "./irt/";
        m_name += fileNames[m_currentModel];
        Vertex size = {0.f,0.f,0.f};
        FileMarshaller fm;
        size = fm.loadFromFile( *m_gpuKernel, m_name, center, 5000.f );
    }
#endif // 0


    // Gallery
#if 1
    float nbPhotos(12.f);
    int nbRand(9);
    FLOAT2 step = {2.f*PI/6,PI/2};

    Vertex position = {0.f,0.f,0.f};
    Vertex size = {18000.f,18000.f,18000.f};
    Vertex frame = {0.05f,0.05f,1.f};
    m_groundHeight = -PI/4*size.y;

    int i(0);
    for( float y(-PI/4.f); y<PI/4.f; y+=step.y)
    {
        for( float x(-PI); x<PI; x+=step.x)
        {
            //if( rand()%4 == 0 )
            {
                int m(60+i%nbRand);

                Vertex p0,p1,p2,p3;
                p0.x = cos(x+frame.x); //cos(y)*cos(x);
                p0.y = y+frame.y; //sin(y);
                p0.z = sin(x+frame.x); //cos(y)*sin(x);

                p1.x = cos(x+step.x-frame.x);
                p1.y = y+frame.y; //sin(y);
                p1.z = sin(x+step.x-frame.x);

                p2.x = cos(x+step.x-frame.x);
                p2.y = y+step.y-frame.y; //sin(y+step.y);
                p2.z = sin(x+step.x-frame.x);

                p3.x = cos(x+frame.x);
                p3.y = y+step.y-frame.y; //sin(y+step.y);
                p3.z = sin(x+frame.x);

                Vertex normal0 = {position.x-p0.x,position.y-p0.y,position.z-p0.z};
                Vertex normal1 = {position.x-p1.x,position.y-p1.y,position.z-p1.z};
                Vertex normal2 = {position.x-p2.x,position.y-p2.y,position.z-p2.z};
                Vertex normal3 = {position.x-p3.x,position.y-p3.y,position.z-p3.z};
                m_nbPrimitives = m_gpuKernel->addPrimitive( ptTriangle);
                m_gpuKernel->setPrimitive( m_nbPrimitives,
                                           position.x+p0.x*size.x, position.y+p0.y*size.y, position.z+p0.z*size.z,
                                           position.x+p1.x*size.x, position.y+p1.y*size.y, position.z+p1.z*size.z,
                                           position.x+p2.x*size.x, position.y+p2.y*size.y, position.z+p2.z*size.z,
                                           0.f, 0.f, 0.f,
                                           m);
                //m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives,false);
                m_gpuKernel->setPrimitiveNormals(m_nbPrimitives, normal0, normal1, normal2 );

                {
                    Vertex tc0 = {1.f,1.f,0.f};
                    Vertex tc1 = {0.f,1.f,0.f};
                    Vertex tc2 = {0.f,0.f,0.f};
                    m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,
                                                                tc0, tc1, tc2);
                }

                m_nbPrimitives = m_gpuKernel->addPrimitive( ptTriangle);
                m_gpuKernel->setPrimitive( m_nbPrimitives,
                                           position.x+p2.x*size.x, position.y+p2.y*size.y, position.z+p2.z*size.z,
                                           position.x+p3.x*size.x, position.y+p3.y*size.y, position.z+p3.z*size.z,
                                           position.x+p0.x*size.x, position.y+p0.y*size.y, position.z+p0.z*size.z,
                                           0.f, 0.f, 0.f,
                                           m);
                //m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives,false);
                m_gpuKernel->setPrimitiveNormals(m_nbPrimitives, normal2, normal3, normal0 );
                {
                    Vertex tc0 = {0.f,0.f,0.f};
                    Vertex tc1 = {1.f,0.f,0.f};
                    Vertex tc2 = {1.f,1.f,0.f};
                    m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,
                                                                tc0, tc1, tc2);
                }
                ++i;
            }
        }
    }
    for( int i(0); i<40; ++i )
    {
        m_nbPrimitives = m_gpuKernel->addPrimitive(  ptSphere );
        m_gpuKernel->setPrimitive(  m_nbPrimitives,
                                    rand()%10000-5000.f,
                                    rand()%10000-5000.f,
                                    rand()%10000-5000.f,
                                    200.f+rand()%200, 0.f, 0.f,
                                    41+rand()%2 );
    }
#else
    int nbPhotos(18);
    int size(30000);
    int halfSize(size/2);

    for( int i(0); i<nbPhotos; ++i)
    {
        Vertex position;
        position.x = static_cast<float>(rand()%size-halfSize);
        position.y = static_cast<float>(rand()%size-halfSize);
        position.z = static_cast<float>(rand()%size);
        // Caustic
        // 1
        float causticSize = 2000.f;
        {
            m_nbPrimitives = m_gpuKernel->addPrimitive( ptTriangle);
            m_gpuKernel->setPrimitive( m_nbPrimitives,
                                       position.x+causticSize, position.y                           , position.z,
                                       position.x-causticSize, position.y                           , position.z,
                                       position.x-causticSize, position.y-m_groundHeight+causticSize, position.z,
                                       0.f, 0.f, 0.f,
                                       i+60);
            Vertex normal = {0.f,0.f,1.f};
            m_gpuKernel->setPrimitiveNormals(m_nbPrimitives,
                                             normal, normal, normal );

            Vertex tc0 = {1.f,1.f,0.f};
            Vertex tc1 = {0.f,1.f,0.f};
            Vertex tc2 = {0.f,0.f,0.f};
            m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,
                                                        tc0, tc1, tc2);
        }

        // 2
        {
            m_nbPrimitives = m_gpuKernel->addPrimitive( ptTriangle);
            m_gpuKernel->setPrimitive( m_nbPrimitives,
                                       position.x-causticSize, position.y-m_groundHeight+causticSize, position.z,
                                       position.x+causticSize, position.y-m_groundHeight+causticSize, position.z,
                                       position.x+causticSize, position.y                           , position.z,
                                       0.f, 0.f, 0.f,
                                       i+60);
            Vertex normal = {0.f,0.f,1.f};
            m_gpuKernel->setPrimitiveNormals(m_nbPrimitives,
                                             normal, normal, normal );
            Vertex tc0 = {0.f,0.f,0.f};
            Vertex tc1 = {1.f,0.f,0.f};
            Vertex tc2 = {1.f,1.f,0.f};
            m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives,
                                                        tc0, tc1, tc2);
        }
    }
#endif
}

void GalleryScene::doAnimate()
{
    m_rotationAngles.y = 0.005f;
    //m_rotationAngles.z = 0.01f;
    m_gpuKernel->rotatePrimitives( m_rotationCenter, m_rotationAngles );
    m_gpuKernel->compactBoxes(false);
}

void GalleryScene::doAddLights()
{
    // Lights
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, 5000.f, 0.f,0,0,0,DEFAULT_LIGHT_MATERIAL);
    m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives,false);
}
