attribute vec4 aPosCoord;
attribute vec4 aColor;
uniform mat4 uMVPMatrix;

varying vec4 vColor;

void main(void)
{
	vColor = aColor;
  	gl_Position = uMVPMatrix * aPosCoord;
	//gl_Position.z = gl_Position.w * clamp(gl_Position.z / gl_Position.w, 0.0, 1.0); //Disable z-clipping
	//gl_Position = vec4(gl_Position.xy /  gl_Position.w, 1.0, 1.0); //Disable z-clipping
}
