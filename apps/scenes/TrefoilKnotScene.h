/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class TrefoilKnotScene :
        public Scene
{

public:
    TrefoilKnotScene( const std::string& name, const int nbMaxPrimitivePerBox  );
    ~TrefoilKnotScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

private:
    void trefoilKnot(float R, float t, FLOAT4& p);
    void torus(float R, float t, FLOAT4& p );
    void star(float R, float t, FLOAT4& p );
    void spring(float R, float t, FLOAT4& p);
    void heart(float R, float u, float v, FLOAT4& p);
    void thing(float R, float t, FLOAT4 a, FLOAT4& p);
    void moebius(float R, float u, float v, float s, float du, float dv, FLOAT4& p );

};

