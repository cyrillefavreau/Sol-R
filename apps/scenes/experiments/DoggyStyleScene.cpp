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

#include "DoggyStyleScene.h"

#ifdef WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

#include <math.h>

DoggyStyleScene::DoggyStyleScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
{
}

DoggyStyleScene::~DoggyStyleScene(void)
{
}

void DoggyStyleScene::doInitialize()
{
    vec3f center = {-3500.f, m_groundHeight + 2500.f, 0.f};
    createDog(center, rand() % 20 + 30, 500.f, 1010);
    center.x = 3500.f;
    createDog(center, rand() % 20 + 30, 500.f, 1011);
    vec4f angles = {0.f, static_cast<float>(M_PI), 0.f};
    m_gpuKernel->rotatePrimitives(center, angles);
}

void DoggyStyleScene::doAnimate()
{
}

void DoggyStyleScene::doAddLights()
{
    // Laser
    for (int i(0); i < 3; ++i)
    {
        vec4f p1, p2;
        p1.x = rand() % 10000 - 5000.f;
        p1.y = 5000.f;
        p1.z = rand() % 10000 - 5000.f;
        p2.x = -(rand() % 10000 - 5000.f);
        p2.y = -5000.f;
        p2.z = -(rand() % 10000 - 5000.f);
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
        m_gpuKernel->setPrimitive(m_nbPrimitives, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, 100.f, 100.f, 100.f,
                                  DEFAULT_LIGHT_MATERIAL - i);
    }
}
