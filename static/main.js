var playTraj = function(option) {
    $('.playButton').attr('disabled', true);
    $.get('/play/' + option).success(function() {
	$('.playButton').attr('disabled', false);
	console.log('Finished playing trajectory.');
    });
};
