var playTraj = function() {
    $('.playButton').attr('disabled', true);
    $.get('/play').success(function() {
	$('.playButton').attr('disabled', false);
	console.log('Finished playing trajectory.');
    });
};
var plotTraj = function() {
    $('.playButton').attr('disabled', true);
    $.get('/plot').success(function() {
	$('.playButton').attr('disabled', false);
	console.log('Finished plotting trajectory.');
    });
};
