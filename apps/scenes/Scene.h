/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#undef USE_LEAPMOTION
#undef USE_SIXENSE

// System
#ifdef WIN32
#include <windows.h>
#endif // WIN32
#include <string>

// Raytracer
#include <GPUKernel.h>

class Scene
{
public:
    Scene( const std::string& name, const int nbMaxPrimitivePerBox );
    virtual ~Scene(void);

public:
    void initialize(const int width, const int height);
    void animate();
    void render( const bool& animate );

    virtual void renderText();

    void createRandomMaterials( bool update, bool lightsOnly );
    void createMoleculeMaterials( bool update=false );
    void loadTextures( const std::string& path, const std::string& filter );

    void addCornellBox( int boxType );

    void saveToFile();
    void loadFromFile(const float scale);

    void createSkeleton();
    void animateSkeleton();

    void rotatePrimitives( Vertex rotationCenter, Vertex angles );

public:
    SceneInfo& getSceneInfo();
    std::string& getName();

    int getCornellBox() { return m_cornellBoxType; }
    void setCornellBox( int b ) { m_cornellBoxType = b; }

    float getGroundHeight() { return m_groundHeight; }
    void setGroundHeight( const float h ) { m_groundHeight = h; }

public:
    //void setMaterialTexture( const int& index, const int& texture );

public:
    int getNbPrimitives() { return m_gpuKernel->getNbActivePrimitives(); }
    GPUKernel* getKernel() { return m_gpuKernel; }
    int getNbHDRI() {return m_nbHDRI;}

public:
    void setCurrentModel( const int& currentModel ) { m_currentModel=currentModel; }

protected:
    virtual void doInitialize() = 0;
    virtual void doPostInitialize() {};
    virtual void doAnimate() = 0;
    virtual void doAddLights() = 0;

protected:
    void createWorm( Vertex center, int boxId, int material );
    void createDog( Vertex center, int material, float size, int boxid );

protected:
    std::string m_name;
    GPUKernel* m_gpuKernel;

    // Textures
    int m_nbHDRI;

    // Scene information
    int       m_nbBoxes;
    int       m_cornellBoxType;
    float     m_groundHeight;
    int       m_maxPathTracingIterations;
    int       m_nbMaxPrimitivePerBox;

    // Primitives
    int m_nbPrimitives;
    // Materials
    int m_nbMaterials;

    // Scene models
    int m_currentModel;

protected:
    // Animation
    Vertex m_rotationCenter;
    Vertex m_rotationAngles;

#ifdef USE_KINECT
protected:
    int    m_skeletonPrimitiveIndex;
    float  m_skeletonSize;
    float  m_skeletonThickness;
    Vertex m_skeletonPosition;
    Vertex m_skeletonOldPosition;

    float m_skeletonKinectSpace;
    float m_skeletonKinectSize;
    int   m_skeletonKinectStep;
    int   m_skeletonKinectNbSpherePerBox;
    Vertex m_acceleration;

    Vertex m_previewViewPos;
#endif // USE_KINECT

private:
    Vertex m_modelRotationAngle;
    Vertex m_modelPosition;
    Vertex m_modelTranslation;

};
