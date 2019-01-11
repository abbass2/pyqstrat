//
// This source code resides at www.jaeckel.org/LetsBeRational.7z .
//
// ======================================================================================
// Copyright © 2013-2014 Peter Jäckel.
// 
// Permission to use, copy, modify, and distribute this software is freely granted,
// provided that this notice is preserved.
//
// WARRANTY DISCLAIMER
// The Software is provided "as is" without warranty of any kind, either express or implied,
// including without limitation any implied warranties of condition, uninterrupted use,
// merchantability, fitness for a particular purpose, or non-infringement.
// ======================================================================================
//
#ifndef IMPORTEXPORT_H
#define IMPORTEXPORT_H

#if defined(_WIN32) || defined(_WIN64)
#   define EXPORT __declspec(dllexport)
#   define IMPORT __declspec(dllimport)
# else
#   define EXPORT
#   define IMPORT
#endif

#ifdef __cplusplus
#   define EXTERN_C extern "C"
#else
#   define EXTERN_C
#endif

#   define EXPORT_EXTERN_C EXTERN_C EXPORT
#   define IMPORT_EXTERN_C EXTERN_C IMPORT

#endif // IMPORTEXPORT_H
