#include <rc_usefulincludes.h> 
#include <roboticscape.h>

int speed;

/*******************************************************************************
* void on_pause_released() 
*	
* Make the pause button toggle between paused and running states.
*******************************************************************************/
void on_pause_released(){
	speed = 1000 + rand()%800;
	// Toggle betewen paused and running modes
	if(rc_get_state()==RUNNING)	rc_set_state(PAUSED);
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
	int i = 0;
	const int samples = 100;     // Check for release 100 times in this period
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
	//rc_send_esc_pulse_normalized(1, 0.3);
	rc_send_servo_pulse_us(1, speed);
	printf( "Speed: %d", speed);
	rc_send_servo_pulse_us(2, 1750);
	printf("Moving Forward\n");
}

void mBack() {
	printf("Moving Back\n");
}

void mStop() {
	//rc_send_esc_pulse_normalized(1, 0.0);
	rc_send_servo_pulse_us(1, 1500);
	rc_send_servo_pulse_us(2, 1400);
	printf("Stopping\n");
}

void mLeft() {
	printf("Moivng Left\n");
}

void mRight() {
	printf("Moving Right\n");
}

/*******************************************************************************
* int main() 
*
* - call to rc_initialize() at the beginning
* - main while loop that checks for EXITING condition
* - rc_cleanup() at the end
*******************************************************************************/
int main(){
	printf("\nRunning.\n");
	
	// Initialize cape library first
	if(rc_initialize()){
		fprintf(stderr,"ERROR: failed to initialize rc_initialize(), did you run with sudo?\n");
		return -1;
	}

	rc_enable_servo_power_rail();
	rc_set_pause_pressed_func(&on_pause_pressed);
	rc_set_pause_released_func(&on_pause_released);
	rc_set_state(RUNNING); 
	
	// Keep looping until state changes to EXITING
	while(rc_get_state() != EXITING){
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

	printf("All motors off.\n\n");
	rc_disable_servo_power_rail();
	rc_cleanup(); 
	return 0;
}
