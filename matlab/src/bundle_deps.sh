#!/bin/sh
# bundle_deps.sh - Collect shared library dependencies for MEX deployment
#
# Usage: ./bundle_deps.sh <mex_file> <output_dir>
#
# Copies non-system shared library dependencies into output_dir,
# strips debug symbols, and patches rpath/install_name so libs find
# each other at runtime.
#
# Supports: Linux (.mexa64/.mex), macOS (.mexmaci64/.mexmaca64), Windows (.mexw64)
#
# Place in: matlab/src/bundle_deps.sh

set -e

TARGET="$1"
DEST="$2"

if [ -z "$TARGET" ] || [ -z "$DEST" ]; then
    echo "Usage: $0 <mex_file> <output_dir>"
    exit 1
fi

if [ ! -f "$TARGET" ]; then
    echo "Error: $TARGET not found"
    exit 1
fi

mkdir -p "$DEST"

echo "Collecting dependencies from: $TARGET"
echo "Output directory: $DEST"
echo ""

OS=$(uname -s)

# =========================================================================
# Linux
# =========================================================================
if [ "$OS" = "Linux" ]; then

    ldd "$TARGET" | grep '=>' | awk '{print $3}' | while read lib; do
        [ -z "$lib" ] && continue
        [ ! -f "$lib" ] && continue

        base=$(basename "$lib")

        # Skip system/MATLAB libraries (always present on target)
        case "$base" in
            libmx.so*|libmex.so*)           continue ;;  # MATLAB
            libc.so.*|libm.so.*)             continue ;;  # glibc
            libdl.so*|librt.so*)             continue ;;  # glibc
            libpthread.so*|libgcc_s.so*)     continue ;;  # system runtime
            ld-linux*|linux-vdso*)           continue ;;  # kernel/linker
            *)
                # Skip if already copied (for multi-target bundling)
                if [ ! -f "$DEST/$base" ]; then
                    cp -vL "$lib" "$DEST/"
                fi
                ;;
        esac
    done

    echo ""
    echo "Stripping debug symbols..."
    for f in "$DEST"/*.so*; do
        [ -f "$f" ] && strip --strip-unneeded "$f" 2>/dev/null || true
    done

    echo ""
    if command -v patchelf >/dev/null 2>&1; then
        echo "Patching rpath on bundled libs..."
        for f in "$DEST"/*.so*; do
            if [ -f "$f" ]; then
                patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || true
            fi
        done
        echo "Done. All bundled libs will find each other via \$ORIGIN rpath."
    else
        echo "WARNING: patchelf not found."
        echo "Indirect dependencies (e.g. libamd used by libumfpack) may not"
        echo "resolve from $DEST on other machines."
        echo "Install with: sudo apt install patchelf"
    fi

# =========================================================================
# macOS
# =========================================================================
elif [ "$OS" = "Darwin" ]; then

    otool -L "$TARGET" | tail -n +2 | awk '{print $1}' | while read lib; do
        [ -z "$lib" ] && continue

        base=$(basename "$lib")

        # Skip system frameworks and libraries
        case "$lib" in
            /usr/lib/*)              continue ;;  # system libs
            /System/*)               continue ;;  # system frameworks
            *libSystem*)             continue ;;  # libSystem
            *Accelerate*)            continue ;;  # Accelerate framework
            @rpath/libmx.*)          continue ;;  # MATLAB
            @rpath/libmex.*)         continue ;;  # MATLAB
        esac

        # Resolve @rpath, @loader_path references
        reallib="$lib"
        case "$lib" in
            @rpath/*|@loader_path/*|@executable_path/*)
                # Try to find in common locations
                shortname=$(echo "$lib" | sed 's|@rpath/||;s|@loader_path/||;s|@executable_path/||')
                for searchdir in /opt/homebrew/lib /usr/local/lib \
                    /opt/homebrew/opt/suite-sparse/lib \
                    /opt/homebrew/opt/openblas/lib; do
                    if [ -f "$searchdir/$shortname" ]; then
                        reallib="$searchdir/$shortname"
                        break
                    fi
                done
                ;;
        esac

        [ ! -f "$reallib" ] && continue

        if [ ! -f "$DEST/$base" ]; then
            cp -vL "$reallib" "$DEST/"
            # Make writable for install_name_tool
            chmod u+w "$DEST/$base"
        fi
    done

    echo ""
    echo "Stripping debug symbols..."
    for f in "$DEST"/*.dylib*; do
        [ -f "$f" ] && strip -x "$f" 2>/dev/null || true
    done

    echo ""
    echo "Patching install names..."
    for f in "$DEST"/*.dylib*; do
        [ ! -f "$f" ] && continue
        base=$(basename "$f")

        # Set the library's own install name to @loader_path
        install_name_tool -id "@loader_path/blqmr_deps/$base" "$f" 2>/dev/null || true

        # Patch all dependencies to point to @loader_path
        otool -L "$f" | tail -n +2 | awk '{print $1}' | while read dep; do
            depbase=$(basename "$dep")
            if [ -f "$DEST/$depbase" ]; then
                install_name_tool -change "$dep" "@loader_path/$depbase" "$f" 2>/dev/null || true
            fi
        done
    done
    echo "Done."

# =========================================================================
# Windows (MSYS2/MinGW/Git Bash)
# =========================================================================
elif echo "$OS" | grep -qiE 'mingw|msys|cygwin'; then

    # On Windows, DLLs next to the .mexw64 are found automatically.
    # Use ldd (MSYS2) or ntldd to find dependencies.
    if command -v ntldd >/dev/null 2>&1; then
        LDDCMD="ntldd -R"
    else
        LDDCMD="ldd"
    fi

    $LDDCMD "$TARGET" 2>/dev/null | grep '=>' | awk '{print $3}' | while read lib; do
        [ -z "$lib" ] && continue
        [ ! -f "$lib" ] && continue

        base=$(basename "$lib")

        # Skip Windows system DLLs and MATLAB DLLs
        case "$base" in
            ntdll.dll|kernel32.dll|KERNEL32.DLL)     continue ;;
            msvcrt.dll|MSVCRT.DLL|ucrtbase.dll)      continue ;;
            api-ms-*|ext-ms-*)                       continue ;;
            libmx.dll|libmex.dll)                    continue ;;
            *)
                if [ ! -f "$DEST/$base" ]; then
                    cp -vL "$lib" "$DEST/"
                fi
                ;;
        esac
    done

    echo ""
    echo "Stripping debug symbols..."
    for f in "$DEST"/*.dll; do
        [ -f "$f" ] && strip --strip-unneeded "$f" 2>/dev/null || true
    done
    echo "Done. Place DLLs next to the .mexw64 file or in blqmr_deps/ on PATH."

else
    echo "Unsupported platform: $OS"
    exit 1
fi