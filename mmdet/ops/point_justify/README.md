### function

用CUDA实现点在多边形内外的判断，采用的是射线法判定交点个数。

源码借鉴自aerialdet的部分代码，并修复了原程序的边界条件不合适导致的错误判断问题。

### usage

见te
st.py，支持多points和多ploygons。