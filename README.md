# CETgraph version 0.1 #
_First public release by_ **Sanli, 14/11/2017**


In its completed form, this package, should enable identifying thermally-diffusing or electro-actuated single (nano)particles (most probably measured in one of the nanoCET setups), track their position and use this tracking information to identify these particles and their temporal dynamics.

Further information about nanoCET: www.nano-EPics.com
## Project Goals ##

*[x] Identify presence of particles and extract their tracks as they pass through the field of view.  
*[x] Use tracking information (position, intensity, point spread function) to determine  diffusion constants and optical scattering cross section of individual particle and distribution of their properties.
*[x] Where beneficial, can use Trackpy standard package
*[ ] Generate presentable reports of particle size distribution or other characterized properties.
*[ ] For electro-actuated particles determine the mobility of the particles.

## Directories ##
* **tracking**: the core classes used for analysis
* **analyzing**: actual code used for analyzing an specific type of datafile
* **presenting**: all examples and templates that can be used for presenting extracted information
* **users**: file exchange between active users
* **examples**: sandbox examples to start
* **upstream_tests**: standardized test to assure backward compatiblity when the corse classes are updated


_For a list of contributors see_ ./AUTHORS.md 
