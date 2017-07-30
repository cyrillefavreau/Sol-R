include_directories(
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/apps 
	${PROJECT_SOURCE_DIR}/generated)

ADD_EXECUTABLE(
  solrViewer
  solrViewer.cpp
  common/Types.h
  common/Utils.cpp
  common/Utils.h
  scenes/Scene.cpp
  scenes/Scene.h
  scenes/animation/KinectFaceTrackingScene.cpp
  scenes/animation/KinectFaceTrackingScene.h
  scenes/animation/MetaballsScene.cpp
  scenes/animation/MetaballsScene.h
  scenes/animation/AnimationScene.cpp
  scenes/animation/AnimationScene.h
  scenes/animation/WaterScene.cpp
  scenes/animation/WaterScene.h
  scenes/maths/TrefoilKnotScene.cpp
  scenes/maths/TrefoilKnotScene.h
  scenes/maths/FractalsScene.cpp
  scenes/maths/FractalsScene.h
  scenes/maths/FractalScene.cpp
  scenes/maths/FractalScene.h
  scenes/experiments/TransparentScene.cpp
  scenes/experiments/TransparentScene.h
  scenes/experiments/XmasScene.cpp
  scenes/experiments/XmasScene.h
  scenes/experiments/GraphScene.cpp
  scenes/experiments/GraphScene.h
  scenes/experiments/Year2013.cpp
  scenes/experiments/Year2013.h
  scenes/experiments/CylinderScene.cpp
  scenes/experiments/CylinderScene.h
  scenes/experiments/GalleryScene.cpp
  scenes/experiments/GalleryScene.h
  scenes/experiments/PerpetualMotionScene.cpp
  scenes/experiments/PerpetualMotionScene.h
  scenes/experiments/DoggyStyleScene.cpp
  scenes/experiments/DoggyStyleScene.h
  scenes/experiments/CornellBoxScene.cpp
  scenes/experiments/CornellBoxScene.h
  scenes/experiments/CubesScene.cpp
  scenes/experiments/CubesScene.h
  scenes/meshes/TrianglesScene.cpp
  scenes/meshes/TrianglesScene.h
  scenes/meshes/ObjScene.cpp
  scenes/meshes/ObjScene.h
  scenes/games/SpindizzyScene.cpp
  scenes/games/SpindizzyScene.h
  scenes/science/SwcScene.cpp
  scenes/science/SwcScene.h
  scenes/science/ColumnScene.cpp
  scenes/science/ColumnScene.h
  scenes/science/MoleculeScene.cpp
  scenes/science/MoleculeScene.h
  )


include_directories(${PROJECT_SOURCE_DIR}/solr)

TARGET_LINK_LIBRARIES(
    solrViewer
    solr
    ${LEAP_LIBRARY}
)

# ------------------------------------------------------------
INSTALL(TARGETS solrViewer DESTINATION bin)
# ------------------------------------------------------------
