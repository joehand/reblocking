/* Builds the minified js file*/
({
    out: "../build/app.min.js",
    name: 'app',
    optimize: "uglify2",
    findNestedDependencies: true,
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
})
