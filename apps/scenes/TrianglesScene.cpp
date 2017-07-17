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

#define _CRT_SECURE_NO_WARNINGS
#ifdef WIN32
#include <windows.h>
#else
#include <algorithm>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#endif // WIN32

#include <Consts.h>
#include <Logging.h>
#include <time.h>

#include <io/FileMarshaller.h>
#include <io/OBJReader.h>
#include <io/PDBReader.h>

#include <iostream>

#include "TrianglesScene.h"

TrianglesScene::TrianglesScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : m_frameIndex(0)
    , Scene(name, nbMaxPrimitivePerBox)
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

    std::string fullFilter("../medias/irt/*.irt");
    hFind = FindFirstFile(fullFilter.c_str(), &FindData);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            if (strlen(FindData.cFileName) != 0)
                fileNames.push_back(FindData.cFileName);
        } while (FindNextFile(hFind, &FindData));
    }
#else
    std::string path = "../medias/irt";
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
            if (strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0)
            {
                std::string fullPath(path);
                fullPath += "/";
                fullPath += dirp->d_name;
                LOG_INFO(1, "Model: " << fullPath);
                fileNames.push_back(fullPath);
            }
        }
        closedir(dp);
    }
#endif // WINew
    if (fileNames.size() != 0)
    {
        float objectScale = 1.f;
        m_currentModel = m_currentModel % fileNames.size();
#ifdef WIN32
        m_name = "../medias/irt/";
        m_name += fileNames[m_currentModel];
#else
        m_name = fileNames[m_currentModel];
#endif // WIN32
        Vertex size = {0.f, 0.f, 0.f};
        Vertex center = {0.f, 0.f, 0.f};
        FileMarshaller fm;
        size = fm.loadFromFile(*m_gpuKernel, m_name, center, objectScale * 5000.f);
        m_groundHeight = -2500.f;
    }
}

void TrianglesScene::doAnimate()
{
    const int nbFrames = 120;
    m_rotationAngles.y = static_cast<float>(-2.f * M_PI / nbFrames);
    m_gpuKernel->rotatePrimitives(m_rotationCenter, m_rotationAngles);
    m_gpuKernel->compactBoxes(false);
}

void TrianglesScene::doAddLights()
{
    // lights
    if (m_gpuKernel->getNbActiveLamps() == 0)
    {
        LOG_INFO(1, "Adding sun light");
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
        m_gpuKernel->setPrimitive(m_nbPrimitives, 5000.f, 5000.f, -5000.f, 10.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
        m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives, false);
    }
}
