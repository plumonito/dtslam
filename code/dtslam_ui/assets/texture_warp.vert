attribute vec4 aPosCoord;
attribute vec2 aTexCoord;
uniform mat4 uMVPMatrix;

varying vec2 vTexCoord;

void main(void)
{
	vTexCoord = aTexCoord;
  	gl_Position = uMVPMatrix * aPosCoord;
}
