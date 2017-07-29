#include <rc_usefulincludes.h> 
#include <roboticscape.h>

/*******************************************************************************
* void on_pause_released() 
*	
* Make the pause button toggle between paused and running states.
*******************************************************************************/
void on_pause_released(){
	// Toggle betewen paused and running modes
	if(rc_get_state()==RUNNING)		rc_set_state(PAUSED);
	else if(rc_get_state()==PAUSED)	rc_set_state(RUNNING);
	return;
}

/*******************************************************************************
* void on_pause_pressed() 
*
* If the user holds the pause button for 2 seconds, set state to exiting, which 
* triggers the rest of the program to exit cleanly.
*******************************************************************************/
void on_pause_pressed(){
	int i=0;
	const int samples = 100;	 // Check for release 100 times in this period
	const int us_wait = 2000000; // 2 seconds
	
	// Keep checking to see if the button is still held down
	for(i=0; i<samples; i++) {
		rc_usleep(us_wait/samples);
		if(rc_get_pause_button() == RELEASED) return;
	}
	printf("Long press detected, shutting down\n");
	rc_set_state(EXITING);
	return;
}

void mForward() {
	rc_set_motor_all(1.0);
	printf("Moving Forward/n");
}

void mBack() {
	rc_set_motor_all(-1.0);
	printf("Moving Back/n");
}

void mStop() {
	rc_set_motor_all(0.0);
	printf("Stopping");
}

void mLeft() {
	rc_set_motor_all(0.0);
	usleep(500000);
	rc_set_motor(1,1);
	rc_set_motor(2,-1);
	printf("Moivng Left");
}

void mRight() {
	rc_set_motor_all(0.0);
	usleep(500000);
	rc_set_motor(1,-1);
	rc_set_motor(2,1);
	printf("Moving Right");
}

/*******************************************************************************
* int main() 
*
* This template main function contains these critical components
* - call to rc_initialize() at the beginning
* - main while loop that checks for EXITING condition
* - rc_cleanup() at the end
*******************************************************************************/
int main(){
	// Initialize cape library first
	if(rc_initialize()){
		fprintf(stderr,"ERROR: failed to initialize rc_initialize(), did you run with sudo?\n");
		return -1;
	}

	printf("\nHello BeagleBone\n");
	rc_set_pause_pressed_func(&on_pause_pressed);
	rc_set_pause_released_func(&on_pause_released);

	rc_set_state(RUNNING); 
	
	// Take H-bridges out of standby
	rc_enable_motors(); 

	// Keep looping until state changes to EXITING
	while(rc_get_state()!=EXITING){
		if(rc_get_state()==RUNNING){
			rc_set_led(GREEN, ON);
			rc_set_led(RED, OFF);
			mForward();
		}
		else if(rc_get_state()==PAUSED){
			rc_set_led(GREEN, OFF);
			rc_set_led(RED, ON);
			mStop();
		}
		usleep(100000);
	}

	// User must have existed, put H-bridges into standby
	rc_disable_motors();	
	printf("All motors off.\n\n");
	rc_cleanup(); 
	return 0;
}
