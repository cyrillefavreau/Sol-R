#pragma once

#ifdef LOGGING
#include <ETWLoggingModule.h>
#include <ETWResources.h>
#else
#include <sstream>
#include <windows.h>

// std::cout << __s.str() << std::endl; \

#define LOG_INFO( __level, __msg ) \
   { \
      if( m_activeLogging ) \
      { \
         std::stringstream __s; \
         __s << "[INFO]  " << __msg; \
         OutputDebugString(__s.str().c_str()); \
      } \
   }

#define LOG_ERROR( __msg ) \
   { \
      std::stringstream __s; \
      __s << "[ERROR] " << __msg; \
      OutputDebugString(__s.str().c_str()); \
   }
#endif
