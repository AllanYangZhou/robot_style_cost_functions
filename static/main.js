var playTraj = function() {
    $('.playButton').attr('disabled', true);
    $.get('/play').success(function() {
	$('.playButton').attr('disabled', false);
	console.log('Finished playing trajectory.');
    });
};

var testModel = function() {
    $.get('/test_model');
};
