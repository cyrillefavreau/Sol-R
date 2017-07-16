/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class FractalsScene : public Scene
{

public:
    FractalsScene( const std::string& name, const int nbMaxPrimitivePerBox  );
    ~FractalsScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

private:
    void createFractals( float maxIterations, FLOAT4 center, int material );
    FLOAT4 MandelBox(Vertex V, Vertex Scale, float R, float S, float C);
    float DE(Vertex pos, const int iterations, const FLOAT2 params);
    bool isSierpinskiCarpetPixelFilled(int i, INT3 v);

};

