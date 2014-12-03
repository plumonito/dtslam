attribute vec4 aPosCoord;
attribute vec4 aColor;
attribute vec3 aNormal;
uniform mat4 uMVPMatrix;

varying vec4 vColor;
varying vec3 vNormal;
varying vec4 vVertexPos;

void main(void)
{
	vColor = aColor;
	vNormal = aNormal;
	vVertexPos = aPosCoord;
  	gl_Position = uMVPMatrix * aPosCoord;
}
