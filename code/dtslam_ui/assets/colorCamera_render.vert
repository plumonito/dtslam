uniform mat3 uCameraK;
uniform vec2 uCameraDist;
uniform float uCameraMaxRadiusSq;
uniform mat4 uRt;

attribute vec4 aPosCoord;
attribute vec4 aColor;
uniform mat4 uMVPMatrix;

varying vec4 vColor;

void main(void)
{
	vColor = aColor;
	
	//To camera coordinates
	vec4 xc = uRt*aPosCoord;
	
	//Normalize
	vec2 xn = xc.xy / xc.z;
	
	//Distort
	float r2 = min(uCameraMaxRadiusSq, dot(xn,xn));
	float r4 = r2*r2;
	float factor = 1.0f + uCameraDist[0]*r2 + uCameraDist[1]*r4;	
	vec2 xd = factor*xn;
	
	//To image coordinates
	vec3 uv = uCameraK*vec3(xd,1);
	
	//Add depth again
	vec3 uvz = xc.z * uv;
	
	//To vec4
	vec4 uv4 = vec4(uvz,1);
	
	//To gl coordinates
	gl_Position = uMVPMatrix * uv4;
}
