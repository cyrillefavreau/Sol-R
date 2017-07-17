/* Copyright (c) 2011-2014, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of Sol-R <https://github.com/cyrillefavreau/Sol-R>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#endif // WIN32
#include <Logging.h>
#include <iostream>

#include <Consts.h>
#include <io/FileMarshaller.h>
#include <io/OBJReader.h>
#include <io/PDBReader.h>

#include "MoleculeScene.h"

MoleculeScene::MoleculeScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
{
    m_groundHeight = -5000.f;
}

MoleculeScene::~MoleculeScene(void)
{
}

void MoleculeScene::doInitialize()
{
    // initialization
    int geometryType(rand() % 5);
    LOG_INFO(1, "Geometry type: " << geometryType);
    int atomMaterialType(0);
    float defaultAtomSize(100.f);
    float defaultStickSize(10.f);
    bool loadModels(true);

    std::vector<std::string> proteinNames;
#ifdef WIN32
    // Proteins vector
    HANDLE hFind(nullptr);
    WIN32_FIND_DATA FindData;

    std::string fullFilter("../medias/pdb/*.pdb");
    hFind = FindFirstFile(fullFilter.c_str(), &FindData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (strlen(FindData.cFileName) != 0)
            {
                std::string shortName(FindData.cFileName);
                shortName = shortName.substr(0, shortName.rfind(".pdb"));
                proteinNames.push_back(shortName);
            }
        } while (FindNextFile(hFind, &FindData));
    }
#else
    std::string path = "../medias/pdb";
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(path.c_str())) == NULL)
    {
        LOG_ERROR(errno << " opening " << path);
    }
    else
    {
        while ((dirp = readdir(dp)) != NULL)
        {
            std::string filename(dirp->d_name);
            if (filename != "." && filename != ".." && filename.find(".pdb") != std::string::npos &&
                filename.find(".mtl") == std::string::npos)
            {
                filename = filename.substr(0, filename.find(".pdb"));
                std::string fullPath(path);
                fullPath += "/";
                fullPath += filename;
                proteinNames.push_back(fullPath);
            }
        }
        closedir(dp);
    }
#endif // WIN32

    if (proteinNames.size() != 0)
    {
        m_currentModel = m_currentModel % proteinNames.size();
        Vertex scale = {200.f, 200.f, 200.f};
        std::string fileName;

        // Scene
        m_name = proteinNames[m_currentModel];

        // PDB
        PDBReader pdbReader;
#ifdef WIN32
        fileName = "../medias/pdb/";
#endif // WIN32
        fileName += proteinNames[m_currentModel];
        fileName += ".pdb";
        Vertex objectSize =
            pdbReader.loadAtomsFromFile(fileName, *m_gpuKernel, static_cast<GeometryType>(geometryType),
                                        defaultAtomSize, defaultStickSize, atomMaterialType, scale, loadModels);

        float size(1.1f);
        objectSize.x *= size;
        objectSize.y *= size;
        objectSize.z *= size;
        if (loadModels)
        {
            fileName = "";
#ifdef WIN32
            fileName += "../medias/pdb/";
#endif // WIN32
            fileName += proteinNames[m_currentModel];
            fileName += ".obj";
            Vertex center = {0.f, 0.f, 0.f};
            OBJReader objReader;
            CPUBoundingBox aabb;
            CPUBoundingBox inAABB;
            objReader.loadModelFromFile(fileName, *m_gpuKernel, center, true, objectSize, true, 1000, false, true, aabb,
                                        false, inAABB);
        }
    }
    FileMarshaller fm;
    fm.saveToFile(*m_gpuKernel, "Molecule.irt");
}

void MoleculeScene::doAnimate()
{
    const int nbFrames = 120;
    m_rotationAngles.y = static_cast<float>(-2.f * M_PI / nbFrames);
    m_gpuKernel->rotatePrimitives(m_rotationCenter, m_rotationAngles);
    m_gpuKernel->compactBoxes(false);
}

void MoleculeScene::doAddLights()
{
    // Lights
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -5000.f, 5000.f, -15000.f, 1.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
    m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives, false);
}
