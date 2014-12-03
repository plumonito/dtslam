attribute vec2 aPosCoord;
uniform mat4 uMVPMatrix;

void main(void)
{
  gl_Position = vec4(aPosCoord.x, aPosCoord.y, 0.0, 1.0) * uMVPMatrix;
}
