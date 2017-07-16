/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#define _CRT_SECURE_NO_WARNINGS
#ifdef WIN32
#include <windows.h>
#else
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <dirent.h>
#include <errno.h>
#endif // WIN32

#include <time.h>
#include <Consts.h>
#include <Logging.h>

#include <io/OBJReader.h>
#include <io/PDBReader.h>
#include <io/FileMarshaller.h>

#include <iostream>

#include "TrianglesScene.h"

TrianglesScene::TrianglesScene( const std::string& name, const int nbMaxPrimitivePerBox  )
    : m_frameIndex(0),Scene( name, nbMaxPrimitivePerBox )
{
    m_currentModel = 0;
    m_groundHeight = -2500.f;
}

TrianglesScene::~TrianglesScene(void)
{
}

void TrianglesScene::doInitialize()
{
    std::vector<std::string> fileNames;
#ifdef WIN32
    // filename vector
    HANDLE hFind(nullptr);
    WIN32_FIND_DATA FindData;

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
#else
    std::string path="./irt";
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
            if(strcmp(dirp->d_name,".")!=0 && strcmp(dirp->d_name,"..")!=0)
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
#endif // WINew
    if( fileNames.size() != 0 )
    {
        float objectScale=1.f; // EPFL 3.0
        m_currentModel=m_currentModel%fileNames.size();
#ifdef WIN32
        m_name = "./irt/";
        m_name += fileNames[m_currentModel];
#else
        m_name = fileNames[m_currentModel];
#endif // WIN32
        Vertex size = {0.f,0.f,0.f};
        // Vertex center = { 0.f, -600.f, 900.f }; EPFL
        Vertex center = { 0.f, 0.f, 0.f };
        FileMarshaller fm;
        size = fm.loadFromFile( *m_gpuKernel, m_name, center, objectScale*5000.f );
        //m_groundHeight = -size.y*objectScale-EPSILON;
        m_groundHeight = -2500.f;

        //m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere ); m_gpuKernel->setPrimitive( m_nbPrimitives, 0.f, 0.f, 0.f, 2200.f, 0.f, 0.f, BASIC_REFLECTION_MATERIAL_001); m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );
    }
    
#if 0
    // initialization
   int   geometryType(gtAtomsAndSticks);
    LOG_INFO(1,"Geometry type: " << geometryType );
    int   atomMaterialType(0);
    float defaultAtomSize(100.f);
    float defaultStickSize(10.f);
    bool loadModels(true);

    std::vector<std::string> proteinNames;
#ifdef WIN32
    // Proteins vector
    fullFilter = "./pdb/*.pdb";
    hFind = FindFirstFile(fullFilter.c_str(), &FindData);
    if( hFind != INVALID_HANDLE_VALUE )
    {
        do
        {
            if( strlen(FindData.cFileName) != 0 )
            {
                std::string shortName(FindData.cFileName);
                shortName = shortName.substr(0,shortName.rfind(".pdb"));
                proteinNames.push_back(shortName);
            }
        }
        while (FindNextFile(hFind, &FindData));
    }
#else
    std::string path="./pdb";
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
                filename.find(".pdb") != std::string::npos && filename.find(".mtl") == std::string::npos )
            {
                filename = filename.substr(0,filename.find(".pdb"));
                std::string fullPath(path);
                fullPath += "/";
                fullPath += filename;
                proteinNames.push_back(fullPath);
            }
        }
        closedir(dp);
    }
#endif // WIN32

    if( proteinNames.size() != 0 )
    {
        m_currentModel=m_currentModel%proteinNames.size();
        Vertex scale = {50.f,50.f,50.f};
        std::string fileName;

        // Scene
        m_name = proteinNames[m_currentModel];

        // PDB
        PDBReader pdbReader;
#ifdef WIN32
        fileName = "./pdb/";
#endif // WIN32
        fileName += proteinNames[m_currentModel];
        fileName += ".pdb";
      //Vertex objectSize = { 5000,5000,5000 };
        Vertex objectSize = pdbReader.loadAtomsFromFile( fileName, *m_gpuKernel, static_cast<GeometryType>(geometryType), defaultAtomSize, defaultStickSize, atomMaterialType, scale, loadModels );

        float size(1.f);
        objectSize.x *= size;
        objectSize.y *= size;
        objectSize.z *= size;
        if( loadModels )
        {
            fileName = "";
#ifdef WIN32
            fileName += "./pdb/";
#endif // WIN32
            fileName += proteinNames[m_currentModel];
            fileName += ".obj";
            Vertex center={0.f,0.f,0.f};
            OBJReader objReader;
            CPUBoundingBox aabb;
            CPUBoundingBox inAABB;
            objReader.loadModelFromFile(fileName, *m_gpuKernel, center, true, objectSize, true, 42, false, true, aabb, false, inAABB);
        }
    }
#endif
}

void TrianglesScene::doAnimate()
{
    const int nbFrames=120;
#if 0
    if(m_frameIndex<nbFrames)
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[255];
        time(&rawtime);
        timeinfo=localtime(&rawtime);
        strftime(buffer,255,"%Y-%m-%d_%H-%M-%S_",timeinfo);
        std::string filename("E:/Cloud/SkyDrive/Samsung Link/Videos/CudaRayTracer_");
        filename+=buffer;
        sprintf(buffer,"%d.jpg",m_frameIndex);
        filename+=buffer;
        m_gpuKernel->generateScreenshot(filename,512,512,m_gpuKernel->getSceneInfo().maxPathTracingIterations.x);
        ++m_frameIndex;
    }
#endif // 0
    m_rotationAngles.y = static_cast<float>(-2.f*M_PI/nbFrames);
    m_gpuKernel->rotatePrimitives( m_rotationCenter, m_rotationAngles );
    m_gpuKernel->compactBoxes(false);
}

void TrianglesScene::doAddLights()
{
    // lights
    if( m_gpuKernel->getNbActiveLamps()==0 )
    {
        LOG_INFO(1,"Adding sun light");
        m_nbPrimitives = m_gpuKernel->addPrimitive( ptSphere );
        m_gpuKernel->setPrimitive( m_nbPrimitives, 5000.f, 5000.f, -5000.f, 10.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
        m_gpuKernel->setPrimitiveIsMovable( m_nbPrimitives, false );
    }
}
