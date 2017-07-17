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
#endif // WIN32
#include <Consts.h>
#include <io/FileMarshaller.h>

#include "CausticScene.h"

CausticScene::CausticScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
{
    m_groundHeight = -2500.f;
}

CausticScene::~CausticScene(void)
{
}

void CausticScene::doInitialize()
{
#ifdef WIN32
    // Load model
    HANDLE hFind(nullptr);
    WIN32_FIND_DATA FindData;

    std::vector<std::string> fileNames;
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
    Vertex center = {0.f, 0.f, -2000.f};
    if (fileNames.size() != 0)
    {
        m_currentModel = m_currentModel % fileNames.size();
        m_name = "../medias/irt/";
        m_name += fileNames[m_currentModel];
        Vertex size = {0.f, 0.f, 0.f};
        FileMarshaller fm;
        size = fm.loadFromFile(*m_gpuKernel, m_name, center, 5000.f);
    }
#endif // WIN32

    int material = 114;
    int causticMaterial = 114;
    // Caustic
    // 1
    float causticSize = 1000.f;
    {
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptTriangle);
        m_gpuKernel->setPrimitive(m_nbPrimitives, causticSize, 0.f, 0.f, -causticSize, 0.f, 0.f, -causticSize,
                                  -m_groundHeight + causticSize, 0.f, 0.f, 0.f, 0.f, causticMaterial);
        Vertex normal = {0.f, 0.f, 1.f};
        m_gpuKernel->setPrimitiveNormals(m_nbPrimitives, normal, normal, normal);

        Vertex tc0 = {1.f, 1.f, 0.f};
        Vertex tc1 = {0.f, 1.f, 0.f};
        Vertex tc2 = {0.f, 0.f, 0.f};
        m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives, tc0, tc1, tc2);
    }

    // 2
    {
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptTriangle);
        m_gpuKernel->setPrimitive(m_nbPrimitives, -causticSize, -m_groundHeight + causticSize, 0.f, causticSize,
                                  -m_groundHeight + causticSize, 0.f, causticSize, 0.f, 0.f, 0.f, 0.f, 0.f,
                                  causticMaterial);
        Vertex normal = {0.f, 0.f, 1.f};
        m_gpuKernel->setPrimitiveNormals(m_nbPrimitives, normal, normal, normal);
        Vertex tc0 = {0.f, 0.f, 0.f};
        Vertex tc1 = {1.f, 0.f, 0.f};
        Vertex tc2 = {1.f, 1.f, 0.f};
        m_gpuKernel->setPrimitiveTextureCoordinates(m_nbPrimitives, tc0, tc1, tc2);
    }

    // Frame
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -causticSize, m_groundHeight, 0.f, -causticSize,
                              -m_groundHeight + causticSize, 0.f, 50.f, 0.f, 0.f, material);
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
    m_gpuKernel->setPrimitive(m_nbPrimitives, causticSize, m_groundHeight, 0.f, causticSize,
                              -m_groundHeight + causticSize, 0.f, 50.f, 0.f, 0.f, material);

    m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -causticSize, -m_groundHeight + causticSize, 0.f, causticSize,
                              -m_groundHeight + causticSize, 0.f, 50.f, 0.f, 0.f, material);
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -causticSize, 0.f, 0.f, causticSize, 0.f, 0.f, 50.f, 0.f, 0.f, material);

    // Spheres
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, 2000.f, m_groundHeight, 0.f, 1000.f, 0.f, 800.f, material);
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -2000.f, m_groundHeight, 0.f, 1000.f, 0.f, 800.f, material);

    // Mirror
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptYZPlane);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -9950.f, 0.f, -2000.f, 0.f, 4000.f, 4000.f, material);

    // Cylinder
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -10000.f, 3500.f, 2000.f, 10000.f, 3500.f, 2000.f, 1000.f, 0.f, 0.f,
                              material);
}

void CausticScene::doAnimate()
{
}

void CausticScene::doAddLights()
{
    // Lights
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, 5000.f, 5000.f, -100.f, 50, 0, 50, DEFAULT_LIGHT_MATERIAL);
}
