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

#pragma once

#include <scenes/Scene.h>

class FractalsScene : public Scene
{
public:
    FractalsScene(const std::string& name);
    ~FractalsScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

private:
    void createFractals(float maxIterations, const vec4f& center, int material);
    vec4f MandelBox(vec3f V, const vec3f& Scale, float R, float S, float C);
    float DE(vec3f pos, const int iterations, const vec2f params);
    bool isSierpinskiCarpetPixelFilled(int i, vec3i v);
};
