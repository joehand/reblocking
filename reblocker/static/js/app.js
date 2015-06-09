requirejs.config({
    paths: {
        'jquery'             : 'libs/jquery-2.0.3',
        'underscore'         : 'libs/underscore',
        'backbone'           : 'libs/backbone',
    },
    shim: {
        backbone: {
            deps: ['jquery', 'underscore'],
            exports: 'Backbone'
        },
        underscore: {
            exports: '_'
        },
    }
});

// Load the main app module to start the app
requirejs(["main"]);
