attribute vec2 aPosCoord;
attribute vec4 aColor;
uniform mat4 uMVPMatrix;

varying vec4 vColor;

void main(void)
{
	vColor = aColor;
  	gl_Position = uMVPMatrix * vec4(aPosCoord.x, aPosCoord.y, 1.0, 1.0);
}
