#!/usr/bin/env python3
#
# Resolve Mac OS X 'aliases' by finding where they point to
# Author: Scott H. Hawley
#
# Description:
# Mac OSX aliases are not symbolic links. Trying to read one will probably crash your code.
# Here a few routines to help. Run these to change the filename before trying to read a file.
# Intended to be called from within other python code
#
# Python port modified from https://hints.macworld.com/article.php?story=20021024064107356
#
# Requirements: osascript (AppleScript), platform, subprocess
#
# TODO: - make it work in parallel
#       - security upgrade: shell call will allow untrusted execution if 'path' contains ';'', etc.
#
# NOTE: By default, this only returns the names of the original source files,
#       but if you set convert=True, it will also convert aliases to symbolic links.
#
import subprocess
import platform

# returns true if a file is an OSX alias, false otherwise
def isAlias(path, already_checked_os=False):
    if (not already_checked_os) and ('Darwin' != platform.system()):  # already_checked just saves a few microseconds ;-)
        return False
    line_1='tell application "Finder"'
    line_2='set theItem to (POSIX file "'+path+'") as alias'
    line_3='if the kind of theItem is "alias" then'
    line_4='   return true'
    line_5='else'
    line_6='   return false'
    line_7='end if'
    line_8='end tell'

    cmd = "osascript -e '"+line_1+"' -e '"+line_2+"' -e '"+line_3+"' -e '"+line_4+"' -e '"+line_5+"' -e '"+line_6+"' -e '"+line_7+"' -e '"+line_8+"'"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        line2 = line.decode('UTF-8').replace('\n','')
        #print("line2 = [",repr(line2),"]",sep="")
        if ('true' == line2):
            return True
    retval = p.wait()
    return False


def resolve_osx_alias(path, already_checked_os=False, convert=False):        # single file/path name
    if (not already_checked_os) and ('Darwin' != platform.system()):  # already_checked just saves a few microseconds ;-)
        return path
    line_1='tell application "Finder"'
    line_2='set theItem to (POSIX file "'+path+'") as alias'
    line_3='if the kind of theItem is "alias" then'
    line_4='   get the posix path of (original item of theItem as text)'
    line_5='else'
    line_6='return "'+path+'"'
    line_7 ='end if'
    line_8 ='end tell'
    cmd = "osascript -e '"+line_1+"' -e '"+line_2+"' -e '"+line_3+"' -e '"+line_4+"' -e '"+line_5+"' -e '"+line_6+"' -e '"+line_7+"' -e '"+line_8+"'"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    line = p.stdout.readlines()[0]        # TODO: this breaks if there's any error messages
    source = line.decode('UTF-8').replace('\n','')
    if (convert):
        os.remove(path)
        os.symlink(source, path)
    return source


def resolve_osx_aliases(filelist, convert=False):  # multiple files
    #print("filelist = ",filelist)
    if ('Darwin' != platform.system()):
        return filelist
    outlist = []
    for infile in filelist:
        outlist.append(resolve_osx_alias(infile, already_checked_os=True, convert=convert))
    #print("outlist = ",outlist)
    return outlist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Resolve OSX aliases')
    parser.add_argument('file', help="alias files to resolve", nargs='+')
    args = parser.parse_args()
    outlist = resolve_osx_aliases(args.file)
    print("outlist = ",outlist)
