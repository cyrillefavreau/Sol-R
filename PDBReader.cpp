/* 
* Raytracing Engine
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "Consts.h"
#include "PDBReader.h"

//#define CONNECTIONS

struct Element
{
   std::string name;
   float radius;
   int materialId;
};

struct Atom
{
   int    processed;
   int    id;
   int    index;
   float4 position;
   int    boxId;
   int    materialId;
   int    chainId;
   int    residue;
   bool   isBackbone;
   bool   isWater;
};

struct Connection
{
   int atom1;
   int atom2;
};

const int NB_ELEMENTS = 70;
const float DEFAULT_ATOM_DISTANCE = 100.f;
const float DEFAULT_STICK_DISTANCE = 1.65f;
const Element elements[NB_ELEMENTS] =
{
   { "C"  , 67.f,  1 },
   { "N"  , 56.f,  2 },
   { "O"  , 48.f,  3 },
   { "H"  , 53.f,  4 },
   { "B"  , 87.f,  5 },
   { "F"  , 42.f,  6 },
   { "P"  , 98.f,  7 },
   { "S"  , 88.f,  8 },
   { "V"  ,171.f,  9 },
   { "K"  ,243.f, 10 },
   { "HE" , 31.f, 11 },
   { "LI" ,167.f, 12 },
   { "BE" ,112.f, 13 },
   { "NE" , 38.f, 14 },
   { "NA" ,190.f, 15 },
   { "MG" ,145.f, 16 },
   { "AL" ,118.f, 17 },
   { "SI" ,111.f, 18 },
   { "CL" , 79.f, 19 },
   { "AR" , 71.f, 20 },
   { "CA" ,194.f, 21 },
   { "SC" ,184.f, 22 },
   { "TI" ,176.f, 23 },
   { "CR" ,166.f, 24 },
   { "MN" ,161.f, 25 },
   { "FE" ,156.f, 26 },
   { "CO" ,152.f, 27 },
   { "NI" ,149.f, 28 },
   { "CU" ,145.f, 29 },
   { "ZN" ,142.f, 30 },
   { "GA" ,136.f, 31 },
   { "GE" ,125.f, 32 },
   { "AS" ,114.f, 33 },
   { "SE" ,103.f, 34 },
   { "BR" , 94.f, 35 },
   { "KR" , 88.f, 36 }, 

   // TODO
   { "OD1" , 50.f, 37 },
   { "OD2" , 50.f, 38 },
   { "CG1" , 50.f, 39 }, 
   { "CG2" , 50.f, 40 },
   { "CD1" , 50.f, 41 },
   { "CB"  , 50.f, 42 },
   { "CG"  , 50.f, 43 },
   { "CD"  , 50.f, 44 },
   { "OE1" , 50.f, 45 },
   { "NE2" , 50.f, 46 },
   { "CZ"  , 50.f, 47 },
   { "NH1" , 50.f, 48 },
   { "NH2" , 50.f, 49 },
   { "CD2" , 50.f, 50 },
   { "CE1" , 50.f, 51 },
   { "CE2" , 50.f, 52 },
   { "CE"  , 50.f, 53 },
   { "NZ"  , 50.f, 54 },
   { "OH"  , 50.f, 55 },
   { "CE"  , 50.f, 56 },
   { "ND1" , 50.f, 57 },
   { "ND2" , 50.f, 58 },
   { "OXT" , 50.f, 59 },
   { "OG1" , 50.f, 60 },
   { "NE1" , 50.f, 61 },
   { "CE3" , 50.f, 62 },
   { "CZ2" , 50.f, 63 },
   { "CZ3" , 50.f, 64 },
   { "CH2" , 50.f, 65 },
   { "OE2" , 50.f, 66 },
   { "OG"  , 50.f, 67 },
   { "OE2" , 50.f, 68 },
   { "SD"  , 50.f, 69 },
   { "SG"  , 50.f, 70 }
};

PDBReader::PDBReader(void) : m_nbBoxes(0), m_nbPrimitives(0)
{
}

PDBReader::~PDBReader(void)
{
}

float4 PDBReader::loadAtomsFromFile(
   const std::string& filename,
   GPUKernel& cudaKernel,
   int boxId, int nbMaxBoxes,
   GeometryType geometryType,
   float defaultAtomSize,
   float defaultStickSize,
   int   materialType)
{
   for( int i=boxId; i<NB_MAX_BOXES; ++i )
   {
      cudaKernel.resetBox(i,true);
   }

   float distanceRatio = 2.f; //(geometryType==gtSticks || geometryType==gtAtomsAndSticks) ? 2.f : 1.f;

   std::map<int,Atom> atoms;
   std::vector<Connection> connections;
   float4 minPos = {  100000.f,  100000.f,  100000.f, 0.f };
   float4 maxPos = { -100000.f, -100000.f, -100000.f, 0.f };

   int index(0);
   std::ifstream file(filename.c_str());
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::string value;
         std::getline( file, line );
         if( line.find("ATOM") == 0 /* || line.find("HETATM") == 0 */ )
         {
            // Atom
            Atom atom;
            atom.index = index;
            index++;
            std::string atomName;
            std::string chainId;
            std::string atomCode;
            size_t i(0);
            while( i<line.length() )
            {
               switch(i)
               {
               case 6: //ID
               case 12:
               case 76: // Atom name
               case 22: // ChainID
               case 30: // x
               case 38: // y
               case 46: // z
                  value = "";
                  break;
               case 21: atom.chainId = (int)line.at(i)-64; break;
               case 11: atom.id = static_cast<int>(atoi(value.c_str())); break;
               case 17: atomCode = value; break;
               case 79: atomName = value; break;
               case 26: 
                  atom.residue = static_cast<int>(atoi(value.c_str())); 
                  break;
               case 37: atom.position.x = static_cast<float>(atof(value.c_str())); break;
               case 45: atom.position.y = static_cast<float>(atof(value.c_str())); break;
               case 53: atom.position.z = -static_cast<float>(atof(value.c_str())); break;
               default:
                  if( line.at(i) != ' ' ) value += line.at(i);
                  break;
               }
               i++;
            }

            // Backbone
            atom.isBackbone = (geometryType==gtBackbone && atomCode.length()==1);

            // Material
            atom.materialId = 1;
            i=0;
            bool found(false);
            while( !found && i<NB_ELEMENTS )
            {
               if( atomName == elements[i].name )
               {
                  found = true;
                  switch( materialType )
                  {
                  case 1: 
                     atom.materialId = (atom.chainId%2==0) ? elements[i].materialId%20 : 20; 
                     break;
                  case 2: 
                     atom.materialId = atom.residue%20; 
                     break;
                  default: 
                     atom.materialId = elements[i].materialId%20; 
                     break;
                  }
                  atom.position.w = (geometryType==gtFixedSizeAtoms) ? defaultAtomSize : elements[i].radius;
               }
               ++i;
            }

            if( geometryType!=gtBackbone || atom.isBackbone ) 
            {
               // Compute molecule size
               // min
               minPos.x = (atom.position.x < minPos.x) ? atom.position.x : minPos.x;
               minPos.y = (atom.position.y < minPos.y) ? atom.position.y : minPos.y;
               minPos.z = (atom.position.z < minPos.z) ? atom.position.z : minPos.z;
             
               // max
               maxPos.x = (atom.position.x > maxPos.x) ? atom.position.x : maxPos.x;
               maxPos.y = (atom.position.y > maxPos.y) ? atom.position.y : maxPos.y;
               maxPos.z = (atom.position.z > maxPos.z) ? atom.position.z : maxPos.z;
            
               // add Atom to the list
               atom.processed = 0;
               if( geometryType==gtSticks || (geometryType==gtAtomsAndSticks && atom.residue%2 == 0) ) 
               {
                  atoms[atom.id] = atom;
               }
               else
               {
                  atoms[index] = atom;
               }
            }
         }

#ifdef CONNECTION
         else if( line.find("CONECT") == 0 )
         {
            // CONNECT
            Connection connection;
            int i(0);
            while( i<line.length() )
            {
               switch(i)
               {
               case  7: // Atom 1
               case 11: connection.atom1 = static_cast<int>(atoi(value.c_str())); value = ""; break;
               case 16: connection.atom2 = static_cast<int>(atoi(value.c_str())); break;
               default:
                  if( line.at(i) != ' ' ) value += line.at(i);
                  break;
               }
               i++;
            }
            connections.push_back(connection);
         }
#endif
      }
      file.close();
   }

   float4 size;
   size.x = (maxPos.x-minPos.x);
   size.y = (maxPos.y-minPos.y);
   size.z = (maxPos.z-minPos.z);

   float4 center;
   center.x = (minPos.x+maxPos.x)/2.f;
   center.y = (minPos.y+maxPos.y)/2.f;
   center.z = (minPos.z+maxPos.z)/2.f;

   m_nbPrimitives = 0;

   // Optimizing boxes

   // First pass
   bool success(false);
   int nbMoleculeBoxes = atoms.size()/2;
   //std::cout << atoms.size() << "/" << nbMoleculeBoxes << std::endl;
   int boxSize = static_cast<int>(pow(static_cast<float>(nbMoleculeBoxes /*nbMaxBoxes*/), (1.f/3.f)) - 1);
   float4 boxSteps;
   boxSteps.x = ( maxPos.x - minPos.x ) / boxSize;
   boxSteps.y = ( maxPos.y - minPos.y ) / boxSize;
   boxSteps.z = ( maxPos.z - minPos.z ) / boxSize;

   int boxesPerPrimitives[NB_MAX_BOXES];
   memset(boxesPerPrimitives,0,sizeof(int)*NB_MAX_BOXES);

   // Find position of atom in grid
   std::map<int,Atom>::iterator it = atoms.begin();
   while( it != atoms.end() )
   {
      Atom atom((*it).second);

      int X = static_cast<int>(( atom.position.x - minPos.x ) / boxSteps.x);
      int Y = static_cast<int>(( atom.position.y - minPos.y ) / boxSteps.y);
      int Z = static_cast<int>(( atom.position.z - minPos.z ) / boxSteps.z);
      atom.boxId = X*boxSize*boxSize + Y*boxSize + Z;

      boxesPerPrimitives[atom.boxId]++;

      ++it;
   }
   int nbBoxes = nbMaxBoxes;
#if 0
   std::cout << "Number of boxes needed         : " << nbBoxes << "/" << NB_MAX_BOXES << std::endl;
#endif // 0

   it = atoms.begin();
   while( it != atoms.end() )
   {
      Atom& atom((*it).second);
      if( atom.processed<2 )
      {
         int X = static_cast<int>(( atom.position.x - minPos.x ) / boxSteps.x);
         int Y = static_cast<int>(( atom.position.y - minPos.y ) / boxSteps.y);
         int Z = static_cast<int>(( atom.position.z - minPos.z ) / boxSteps.z);
         atom.boxId = X*boxSize*boxSize + Y*boxSize + Z;
         if( atom.boxId < 0 ) std::cout << ":-(" << std::endl;
         boxesPerPrimitives[atom.boxId]++;

         if( boxesPerPrimitives[atom.boxId]>NB_MAX_PRIMITIVES )
         {
            std::cout << "pas cool: " << boxesPerPrimitives[atom.boxId] << std::endl;
         }
         else
         {
            int nb;
            float radius = atom.position.w;
            float stickradius = defaultStickSize;
            
            if( geometryType==gtSticks || 
              ( geometryType==gtBackbone /* && atom.isBackbone*/) || 
              ( geometryType==gtAtomsAndSticks && atom.chainId%2 == 1) ) 
            {
               int cptMeshes(0);
               float4 meshes[3];
               std::map<int,Atom>::iterator it2 = atoms.begin();
               while( it2 != atoms.end() )
               {
                  if( it2 != it && (*it2).second.processed<2 && ((*it).second.isBackbone==(*it2).second.isBackbone))
                  {
                     Atom& atom2((*it2).second);
                     float4 a;
                     a.x = atom.position.x - atom2.position.x;
                     a.y = atom.position.y - atom2.position.y;
                     a.z = atom.position.z - atom2.position.z;
                     float distance = sqrtf( a.x*a.x + a.y*a.y + a.z*a.z );
                     float stickDistance = (geometryType==gtBackbone && atom2.isBackbone) ? DEFAULT_STICK_DISTANCE*2.f : DEFAULT_STICK_DISTANCE;
                     if( distance < stickDistance )
                     {
                        stickradius = (geometryType==gtBackbone && !atom.isBackbone) ? defaultStickSize*0.2f : defaultStickSize; 

#if 0
                        if( geometryType==gtBackbone )
                        {
                           meshes[cptMeshes].x = atom2.position.x; 
                           meshes[cptMeshes].y = atom2.position.y; 
                           meshes[cptMeshes].z = atom2.position.z; 
                           atom2.processed++;
                           cptMeshes++;
                           if( cptMeshes == 3 )
                           {
                              nb = cudaKernel.addPrimitive( ptTriangle );
                              cudaKernel.setPrimitive( 
                                 nb,
                                 boxId + atom.boxId,
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.z - center.z), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].z - center.z),
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].z - center.z),
                                 0.f, 0.f,0.f,
                                 3, 1, 1 );

                              nb = cudaKernel.addPrimitive( ptTriangle );
                              cudaKernel.setPrimitive( 
                                 nb,
                                 boxId + atom.boxId,
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[0].z - center.z), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[1].z - center.z),
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[2].x - center.x), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[2].y - center.y), 
                                 distanceRatio*DEFAULT_ATOM_DISTANCE*(meshes[2].z - center.z),
                                 0.f, 0.f,0.f,
                                 10, 1, 1 );
                              cptMeshes = 0;
                           }
#else
                        float4 halfCenter;
                        halfCenter.x = (atom.position.x + atom2.position.x)/2.f;
                        halfCenter.y = (atom.position.y + atom2.position.y)/2.f;
                        halfCenter.z = (atom.position.z + atom2.position.z)/2.f;
                        nb = cudaKernel.addPrimitive( ptCylinder );
                        cudaKernel.setPrimitive( 
                           nb,
                           boxId + atom.boxId,
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.x - center.x), 
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.y - center.y), 
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.z - center.z), 
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(halfCenter.x - center.x), 
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(halfCenter.y - center.y), 
                           distanceRatio*DEFAULT_ATOM_DISTANCE*(halfCenter.z - center.z),
                           stickradius, 0.f,0.f,
                           atom.materialId , 1, 1 );
#endif
                     }
                  }
                  it2++;
               }
               radius = stickradius;
            }
            else
            {
               radius *= 4.f; 
            }
            
            //if( geometryType != gtBackbone )
            {
               nb = cudaKernel.addPrimitive( ptSphere );
               cudaKernel.setPrimitive( 
                  nb, 
                  boxId+atom.boxId,
                  distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.x - center.x), 
                  distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.y - center.y), 
                  distanceRatio*DEFAULT_ATOM_DISTANCE*(atom.position.z - center.z), 
                  radius, 0.f, 0.f,
                  atom.materialId, 1, 1 );
            }
         }
      }
      ++it;
   }
   
#if 0
   // Connections
   std::vector<Connection>::iterator itc = connections.begin();
   while( itc != connections.end() )
   {
      Connection& connection(*itc);

      // Add connection
      Atom& atom1 = atoms[connection.atom1];
      Atom& atom2 = atoms[connection.atom2];

      // Add atom
      int nb = cudaKernel.addPrimitive( ptCylinder );
      cudaKernel.setPrimitive( 
         nb, 
         boxId+atom1.boxId,
         DEFAULT_ATOM_DISTANCE*(atom1.position.x - center.x), 
         DEFAULT_ATOM_DISTANCE*(atom1.position.y - center.y), 
         DEFAULT_ATOM_DISTANCE*(atom1.position.z - center.z), 
         DEFAULT_ATOM_DISTANCE*(atom2.position.x - center.x), 
         DEFAULT_ATOM_DISTANCE*(atom2.position.y - center.y), 
         DEFAULT_ATOM_DISTANCE*(atom2.position.z - center.z), 
         50.f/10.f, 0.f,0.f,
         atom1.materialId, 1, 1 );
      
      nb = cudaKernel.addPrimitive( ptSphere );
      cudaKernel.setPrimitive( 
         nb, 
         boxId+atom1.boxId,
         DEFAULT_ATOM_DISTANCE*(atom1.position.x - center.x), 
         DEFAULT_ATOM_DISTANCE*(atom1.position.y - center.y), 
         DEFAULT_ATOM_DISTANCE*(atom1.position.z - center.z), 
         atom1.position.w, 0.f, 0.f,
         atom1.materialId, 1, 1 );
      nb = cudaKernel.addPrimitive( ptSphere );
      cudaKernel.setPrimitive( 
         nb, 
         boxId+atom1.boxId,
         DEFAULT_ATOM_DISTANCE*(atom2.position.x - center.x), 
         DEFAULT_ATOM_DISTANCE*(atom2.position.y - center.y), 
         DEFAULT_ATOM_DISTANCE*(atom2.position.z - center.z), 
         atom2.position.w, 0.f, 0.f,
         atom2.materialId, 1, 1 );

      ++itc;
   }
#endif // 0

   m_nbBoxes = cudaKernel.compactBoxes();
   m_nbPrimitives = static_cast<int>(atoms.size()+connections.size());

#if 0
   std::cout << "-==========================================================-" << std::endl;
   std::cout << "filename: " << filename << std::endl;
   std::cout << "------------------------------------------------------------" << std::endl;
   std::cout << "Number of atoms      : " << atoms.size() << std::endl;
   std::cout << "Number of boxes      : " << m_nbBoxes << std::endl;
   std::cout << "Number of connections: " << connections.size() << std::endl;
   std::cout << "------------------------------------------------------------" << std::endl;
#endif // 0

   return size;
}
