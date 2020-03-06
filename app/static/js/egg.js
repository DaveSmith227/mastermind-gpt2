// give $ to jQuery
(function($) {
    /**==============================================================
     * EASTER EGGS - Modified from https://codepen.io/kersley/pen/vLXKgP
     * ==============================================================
     */
    var characters = {};
    var pressedKeys = [];

    /**
     * Plays the audio clip for a specific easter egg
     * @param  {string} character name of character
     */
    function playAudio(character) {
        setTimeout(function() {
            characters[character].audioFile.play();
            setTimeout(function() {
                toggleShowChar(character);
                updateEggs();
            }, characters[character].audioLength);
        }, 1000);
    }

    /**
     * Toggles the display of an easter egg character
     * @param  {string} character character name
     */
    function toggleShowChar(character) {
        $('body').toggleClass(characters[character].name + '-show');
    }

    /**
     * Add characters' information once page loads
     */
    $(document).ready(function() {
        characters = {
            'guessed': [],
            'jeremy': {
                'name': 'jeremy',
                'nameKeys': "74,69,82,69,77,89",
                'audioFile': $('audio.jeremy')[0],
                'audioLength': 10000,
            },
        };
    });

    /**
     * Track character names being typed
     */
    $(window).keydown(function(e) {

        pressedKeys.push(e.keyCode);

        // loop through characters
        for (var character in characters) {

            // if the "pressedKeys" contains the current character name
            if (pressedKeys.toString().indexOf(characters[character].nameKeys) >= 0) {

                // reset the pressedKeys array
                pressedKeys = [];

                // add "show" class to animate the character in
                toggleShowChar(character);

                if ($.inArray(character, characters.guessed) === -1) {
                    characters.guessed.push(character);
                }

                // wait a second and play the character's audio file
                playAudio(character);
            }
        }

        // reset pressedKeys at 50 keypresses to save memory
        if (pressedKeys.length > 500) { pressedKeys = []; }

    });

})(jQuery);