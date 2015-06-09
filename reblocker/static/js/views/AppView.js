/* ========================================================================
 * AppView file
 * Author: JoeHand
 * ========================================================================
 */

define([
    'backbone',
    'underscore',
    'jquery',
], function (Backbone, _, $) {

    var AppView = Backbone.View.extend({

        events: {

        },

        initialize: function(opt) {

        },

        render: function() {
            console.log(this);
            return this;
        },

    });

    return AppView;
});
